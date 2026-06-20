# Handoff — section/method-level KG (layout-based heading extraction)

Adds **method/section nodes** to the knowledge graph so questions like *"what does Method
504.3 §2.2.2 reference?"* become answerable. Today the KG only has spec-citation nodes
(MIL-STD/ISO/ASTM…) at the **document** level — individual test methods are not nodes, so
`spec_lookup("Method 510.7")` always misses (the agent confirmed this in testing).

## What's in this branch

`backend/app/services/kg_headings.py` — **new, self-contained, no LLM, no schema change.**
- `extract_sections(pdf_bytes) -> List[Heading]` — layout/font heading detector.
- `build_section_spec_map(pdf_bytes) -> (headings, section_specs, doc_specs)` — one pass that
  also attributes each spec ID (via existing `kg_extractor.extract_specs`) to the **section
  active at that line** (line-level, more precise than page-level).

Detection is **adaptive + generic** (validated, see below): a line is a heading if it is
prominent (`size >= body_median+1` OR bold-while-body-is-mostly-non-bold) AND matches a robust
shape — `METHOD n`, `ANNEX X`, or `n(.n){0,3} + title`. Numbered shapes are inherently
**table-immune** (table cells like "Section 1:" / "No." never match `n.n  Title`). TOC pages,
figure/table captions and WARNING/CAUTION callouts are dropped.

`backend/scripts/heading_core.py`, `section_graph_poc.py`, `heading_poc*.py` — PoC/validation
scripts (machine-specific paths; not for production).

## Validation (small sample, in-memory, **zero DB touch**)

Ran `section_graph_poc.py` on slices of MIL-STD-810H:
- **Method 504.3 (p136-155):** full tree (1→1.1; 2→2.1→2.1.1; … 6→6.1/6.2), correct `part_of`,
  annexes tagged `504.3-A/B/C ⟂part_of→ 504.3`, and **54 section→Standard reference edges**, e.g.
  `§2.2.2 ──references──> ASTM B117/D1141/D4814/D975/MIL-PRF-…`, `§6.1 ──references──> <full ref list>`.
- **Methods 510.7 / 500.6 / 514.8:** complete numbered hierarchies, depth ≤4.
- **Table page (Part One Annex A questionnaire):** 0 false sections (28 bold table cells dropped).
- **TOC pages:** skipped.

## Integration — DONE (wired in `kg_pipeline.extract_kg_from_document`)

The structural pass now builds Section nodes from `kg_headings.build_section_spec_map(pdf)` when
`settings.KG_HEADING_SECTIONS` (default **True**) and `document.pdf_path` exists; it falls back to
the old `kg_structure` regex otherwise. **No schema change, no re-ingest** — re-runnable on
already-ingested docs via `POST /api/v1/kg/extract/{id}`. Section canonical_ids are scope-keyed
(`doc:{id}#{h.key}`) so annex-local numbering doesn't collide.

**Verified on the live 1089-page MIL-STD-810H** (`POST /kg/extract/{id}`): the KG went from 1
section to **30 method / 1930 section / 36 annex nodes, 1941 part_of + 830 section→Standard
reference edges**. Spot-checks: `#510.7` and `#504.3` are method nodes with their 1..6 subtree;
`#504.3/2.2.2` (Contaminant Fluid Groups) `--references--> ASTM B117/D1141/D4814/D975/MIL-PRF-…`;
`#504.3-A` (ANNEX A) children are `504.3-A/1 GASOLINE FUELS…`, correctly **not** colliding with the
method's own `504.3/1 SCOPE`.

## Agent reachability — DONE

`agent_tools.section_lookup(name)` resolves a METHOD/section by name or number
("Method 510.7", "510.7"), pulls its whole scope-keyed sub-tree (`doc:ID#510.7` +
`doc:ID#510.7/%`) and returns `referenced_standards` (+ `by_section`). The agent system
prompt routes "what standards does Method/§X reference" to it.

On a section-reference hit, `agent.run_agent` returns a **superset answer = KG + RAG**
(Agent mode is meant to replace RAG mode, so it must contain everything RAG would give
plus the KG enrichment):
  1. **KG block** — deterministic, authoritative: each external standard with the sections
     that cite it (built from `by_section`, NOT from gemma4 synthesis — synthesis
     under-weights the KG block and would otherwise answer ~identically to pure RAG).
  2. **`補充（文件內容檢索）` block** — the normal grounded RAG synthesis (internal
     sections, sequencing, procedural context).

Verified: *"Method 510.7 引用了哪些外部規範?"* → **6 external standards** (ASTM D185-07,
IEC 60068-2-68, MIL-HDBK-310, MIL-STD-210B, MIL-STD-3033, MIL-STD-810) each with citing
sections, PLUS the RAG internal/procedural context. Pure `/rag/query` on the same question
returns **0 external standards** (only internal sections) — confirming the Agent answer is
a strict superset, not RAG-equivalent.

## Still TODO (follow-ups, not blockers)

1. **Spec attribution is line-level** within the document walk (current section = last heading
   seen). Good for reference lists/tables; revisit only if cross-section bleed appears.
3. **Unnumbered headings** (e.g. "INTRODUCTION") are intentionally not emitted (avoids WARNING/
   table-label noise). If a target doc relies on them, add a stricter unnumbered rule or use the
   OCR layout model's `paragraph_title` regions.

## Performance

Heading extraction uses **PyMuPDF (fitz)**: full 1089-page scan ≈ **10s** (an earlier pdfplumber
implementation took ~14 min). The KG structural pass is therefore ~10s; the slow part of a full
KG re-extract remains the LLM spec↔spec relation classification (unchanged).

## Generality

Structure layer (Document/Section/contains/part_of) and the font/bold heading signal are
**doc-type agnostic** and work on any visually-structured PDF. The leaf layer (spec-ID extractor
+ references/supersedes/requires vocab) is still **standards-specific** — for other domains
(contracts/papers) make `kg_extractor` + the relation schema pluggable per `meta.kind`.
