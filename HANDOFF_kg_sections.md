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

## Integration (apply in `kg_pipeline.extract_kg_from_document`, then re-run KG extract)

The downstream node/edge code already exists; only the **source of sections** changes. Replace
the `kg_structure.extract_structure(chunks)` call (regex-on-flattened-text, found ~1 section in
1089 pages) with the PDF-derived map. Re-reads the document's stored PDF; **no schema change,
no re-ingest** — re-runnable on already-ingested docs via `POST /api/v1/kg/extract/{id}`:

```python
# in extract_kg_from_document(), where sections are currently built:
import os
from . import kg_headings
section_specs = {}
sections = []
if getattr(settings, "KG_HEADING_SECTIONS", True) and document.pdf_path and os.path.exists(document.pdf_path):
    with open(document.pdf_path, "rb") as f:
        headings, section_specs, doc_level_specs = kg_headings.build_section_spec_map(f.read())
    sections = [h for h in headings if h.number != "-"]
# then, for each h in sections:
#   sec = kg_service.upsert_entity(db, canonical_id=f"{doc_canon}#{h.number}",
#            type_=h.kind, name=h.title, meta={"number": h.number, "page": h.page, "document_id": document.id})
#   Document --contains--> sec        (if h.parent is None)
#   sec --part_of--> doc_canon#{h.parent}   (if h.parent set; method/annex parent is the method number)
#   for spec in section_specs.get(h.number, ()): sec --references--> spec_entity
# keep the existing Document--references-->doc_level_specs fallback.
```

Also extend the Agent so method/section nodes are reachable: in `agent_tools`, have
`spec_lookup` / a new `section_lookup` resolve `doc:{id}#{number}` and `METHOD n`, and let
`spec_references` traverse section→standard edges.

## Remaining refinements (not blockers)

1. **Annex child-section namespacing.** An annex restarts numbering ("1. GASOLINE FUELS" under
   ANNEX A) which collides with the method's own "1. SCOPE" (both `doc:X#1`). Track "inside annex
   X" during the walk and prefix child sections (e.g. `504.3-A#1`). PoC currently leaves annex
   children at top level.
2. **Spec attribution is line-level within a page** — good for reference lists/tables, but a spec
   mentioned in prose far from its governing heading attaches to the current section, which is the
   intended behaviour. Revisit only if cross-section bleed shows up.
3. **Unnumbered headings** (e.g. "INTRODUCTION") are intentionally not emitted (avoids WARNING
   noise). If a target doc relies on them, add a stricter unnumbered-title rule or use the OCR
   layout model's `paragraph_title` regions.

## Generality

Structure layer (Document/Section/contains/part_of) and the font/bold heading signal are
**doc-type agnostic** and work on any visually-structured PDF. The leaf layer (spec-ID extractor
+ references/supersedes/requires vocab) is still **standards-specific** — for other domains
(contracts/papers) make `kg_extractor` + the relation schema pluggable per `meta.kind`.
