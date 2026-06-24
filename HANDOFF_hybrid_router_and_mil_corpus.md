# Handoff — hybrid query router, serial KG queue, pdfminer OOM fix + MIL corpus ingest

Three backend/frontend features plus a large data ingest. All tested live against a running
backend (port 8001) + frontend (5175). No schema change, no Alembic needed.

---

## 1. Hybrid query router — 3 modes, default "混合(hybrid)"

**Why:** content questions ("what are the values / how is it done") answer *better and faster*
with **pure RAG** (full data tables, ~30-120s). Relationship questions ("what does X cite, who
supersedes it, version chain, 有哪些子項") answer better with the **Agent** (KG tools). The Agent
is NOT a universal superset — on a pure content question it can detour through KG tools, dilute
the answer and run ~2× slower (observed: MIL-HDBK-310 values → RAG 127s with tables vs Agent 285s
with only category thresholds). So we route per-question instead of forcing one mode.

**Classifier** — `services/agent.py::route_mode(question) -> "agent" | "rag"`:
- `_RELATION_RE`: 取代/版本/衍生/引用/參考/關係/誰引用/哪些(規範|標準|文件|method)/跟…關係 / supersede|references|version|… → **agent**
- `_ENUM_STRUCT_RE`: structural enumeration only — 子項目/子測試/列舉/有哪些(測試|方法|程序|章節…) → **agent**. Deliberately **excludes** "哪些數值/資訊/數據" (those are content → rag).
- else → **rag**. (9/9 unit cases + live verified.)

**Endpoint** — `POST /api/v1/agent/route` (`api/v1/agent.py`): SSE. First emits
`event: route {mode}`, then for agent mode streams the normal thought/tool_call/observation/final;
for rag mode runs `agent.run_rag_only()` (same retrieval + grounded synthesis as /rag/query) and
emits a single `final`. Persists to `UserConversation` with `mode="hybrid:<rag|agent>"`.

**Frontend** — `frontend/src/pages/QAConsolePage.jsx`:
- Replaced the 2-state Agent `Switch` with a 3-mode `Segmented`: **純RAG | 混合 | Agent**, default
  `hybrid` (localStorage `qa_mode`; old `qa_agent_mode=1` still maps to `agent`).
- `runRouteStream()` hits `/agent/route`, reuses `postAgentStream` (now takes a `url` arg).
- Each answer shows a badge: `混合→內容查詢(RAG)` / `混合→關係查詢(Agent)`.
- lint clean; UI verified (default selects 混合; content Q auto-routes to RAG with badge).

---

## 2. Serial KG queue — fixes `database is locked` on multi-upload

**Why:** KG extraction is write-heavy and the DB is SQLite (one writer). FastAPI
`BackgroundTasks` run in a threadpool, so ingesting several docs in quick succession spawned
**concurrent** KG tasks → `sqlite3.OperationalError: database is locked` (HTTP 500). Hit hard
during the 38-doc bulk ingest.

**Fix** — `services/kg_queue.py` (**new**): one daemon worker thread + a `queue.Queue`. All KG
extraction is funneled through it, so only **one KG runs at a time, ever**.
- `services/documents.py` auto-KG: was inline `run_kg_extract_task(...)` → now
  `kg_queue.enqueue(task.id, document_id)`.
- `api/v1/kg.py` `POST /kg/extract`: was `background_tasks.add_task(...)` → now `enqueue()`,
  returns `queue_depth`.
- `main.py` starts the worker at boot (`KG serial worker started` in log).

Tasks still appear instantly as `pending` and flip to running/completed as the worker reaches
them. **Stress test:** 5 simultaneous `/kg/extract` → all HTTP 202, queue_depth 1→4, completed
strictly one-at-a-time, zero 500s. → `KG_AUTO_EXTRACT=True` is now safe even for rapid multi-file
uploads; the only cost is queue wait (background, doesn't block RAG which reads via WAL).

---

## 3. pdfminer DEBUG logging → OOM on large PDFs (fix)

`utils/logging_config.py` set the root logger to DEBUG, and pdfminer's LZW/CCITT decoders emit a
DEBUG line **per code** — on a 28 MB PDF (MIL-HDBK-781A) this produced millions of lines and the
backend process was killed mid-`create` (no graceful log = OOM/SIGKILL). Fix: mute
`pdfminer / pdfplumber / PIL / fontTools / fitz` to WARNING always. 781A then ingested fine
(347 chunks). **This was the real cause of the "backend died on big files", not file size.**

---

## 4. Agent system-prompt tweak

`services/agent.py` `_SYSTEM_PROMPT_TEMPLATE`: when the user asks about a specific standard by id
("IEC 60068-2-68 是什麼"), resolve it with `spec_lookup` **first** so rag_search can't confuse it
with a similarly-numbered standard (observed IEC 60068-2-68 ↔ IEC 68-2-52), and state plainly when
the referenced standard's full text isn't in the corpus (titles are, content usually isn't).

---

## 5. Data — 40 MIL/AR standards ingested + KG (not code; DB/FAISS state)

Downloaded the free, public-domain US-military standards referenced by MIL-STD-810H from EverySpec
(+ AR 70-38 from Army Publishing). ASTM/IEC ones are **paywalled** and were NOT ingested.
Now in the corpus (all with content): **40 MIL/AR docs, 7,546 chunks, KG 3,074 entities /
4,972 relations.** Two scanned PDFs (MIL-STD-740-1/-2) were ingested via **GPU OCR** (`force_ocr`).

**Known limitation:** 13 docs (mostly MIL-PRF specs + the 2 OCR'd) have **spec-citation KG
relations but no section-structure nodes** — the layout heading detector (`kg_headings.py`, tuned
for MIL-STD-810H's bold `METHOD` style) can't parse flat spec/scanned layouts. Relationship
queries (who cites whom) work; section-level breakdown is absent for these. Future improvement:
broaden heading detection or accept citations-only for spec sheets.

---

## 6. Deterministic relationship / enumeration answers (agent.py) + over-trigger guards

Free-form LLM synthesis dropped existing KG edges (observed: "what supersedes 210B", "who cites
MIL-PRF-7808", "310 ↔ 210" all lost edges the graph already had). Fix mirrors the proven
`section_lookup` superset pattern — build a **deterministic block from the KG, then add the RAG
synthesis as a supplement**. In `run_agent` post-loop, in priority order:

- **`_build_spec_relation_block(db, canonical_id)`** — 版本家族 (derived from node names, always
  correct direction) + 引用(outgoing) + 被引用(incoming, self-citations filtered) + 版本取代.
  `_pick_spec_center(candidates, question)` picks the spec **named in the question** (so
  "MIL-DTL-901 被誰引用" isn't hijacked when the agent also looked up 810H). A deterministic
  fallback resolves a spec id straight from the question (`_SPEC_ID_RE`) if the agent never called a
  spec tool.
- **method-number section_lookup** — if the question names a method number (`509.7`) for a
  reference question, run `section_lookup` on **that exact number** (LLM was occasionally resolving
  509.7 → 510.7).
- **`list_subitems` for a document now returns its METHOD children** (not all 54 sections) so
  "MIL-STD-810H 有哪些測試方法" lists the 30 methods. `_name_in_query` also matches a doc by its
  paren-stripped core ("MIL-STD-810H (full, plain-text)" ← "MIL-STD-810H …").
- **App-question guard** (`_APP_RE`): "我的設備該做哪些測試…" skips *all* single-entity
  deterministic blocks and goes to synthesis (those questions need multi-method synthesis, e.g.
  shipboard → 528.1 + MIL-STD-167). Enumeration fallback also skips relation questions (`_RELATION_RE`).

## 7. `list_procedures` tool (agent_tools.py)

Procedures (Procedure I/II/…) are **not** KG nodes. New tool scans the method's own text and
returns each `Procedure <roman> — <title>`. Attribution is by the **nearest preceding running
header "METHOD nnn.n"** (NOT page-range slicing — the method's `meta.page` from heading detection
can be a few pages late, which mis-assigned procedures to the next method). Agent routes
"有哪些程序" questions through a deterministic procedure block. Verified: 510.7→Blowing Dust/Sand,
516.8→I-VIII, 501.7→Storage/Operation/Tactical-Standby.

## 8. VL multi-page analysis — dynamic context (ai.py)

`MAX_PDF_ANALYSIS_PAGES=10` but 10 page images ≈ 11k tokens > `OLLAMA_NUM_CTX=8192` → 10-page
analysis 500'd (`exceed_context_size`). Fix: `_vl_num_ctx(n_images)` sizes context to
`min(32768, max(NUM_CTX, 4096 + n*1300))` and is passed via `options={"num_ctx": …}` to both the
single-turn and streaming VL calls (`_chat_with_ollama` gained an `options` param). **Only VL
multi-page calls enlarge the context — RAG/Agent text calls stay at 8192.** Verified 1→10 pages all
work (10 pages: full per-page key points + cross-page summary); 11 pages still rejected (422).

## Validation — two rounds × ~20 test-engineer questions

42 varied questions (procedure / value / relationship / enumeration / version / comparison /
application / honesty) run through `/agent/route`; after the fixes above **42/42 pass** with no
regression on the previously-good section_lookup superset answers. Full Q&A exported for the user.
Honesty cases confirmed: 810H has no Li-battery method (→ MIL-STD-882), EMI/EMC is MIL-STD-461G
(cross-doc, using a newly-ingested standard).

---

## Files changed (for commit)

```
backend/app/services/kg_queue.py        (new — serial KG worker)
backend/app/services/agent.py           (route_mode + run_rag_only + /route helpers; deterministic
                                         spec-relation/procedure blocks; _pick_spec_center; app &
                                         relation guards; method-number section_lookup; prompt)
backend/app/services/agent_tools.py     (list_subitems doc→methods + _name_in_query core match;
                                         new list_procedures tool; section_lookup)
backend/app/api/v1/agent.py             (POST /agent/route)
backend/app/services/ai.py              (_vl_num_ctx + options param on VL chat calls)
backend/app/services/documents.py       (auto-KG → kg_queue.enqueue)
backend/app/api/v1/kg.py                (manual KG → kg_queue.enqueue)
backend/app/main.py                     (start KG worker at boot)
backend/app/utils/logging_config.py     (mute pdfminer/pdfplumber/PIL/fontTools/fitz)
frontend/src/pages/QAConsolePage.jsx    (3-mode Segmented + runRouteStream + route badges)
```
`backend/.env` (gitignored) was toggled during bulk ingest; left at `KG_AUTO_EXTRACT=True`.

**Not for commit** — throwaway machine-specific scripts in `backend/scripts/` (download_mil,
ingest_all_mil, kg_extract_all_mil, fix_empty_docs, fix_supersedes, batch_*, export_excel,
list_procedures-debug, etc.), root-level Playwright `ui_*.mjs`, `.shots/`, `qa_results.jsonl`,
and the `doc_management.db-wal/-shm` files. Source PDFs live outside the repo in
`C:\Users\G635LXG\Downloads\RAG\sample_pdfs\mil\`.

## Ops notes
- One-off DB cleanups already applied to the live DB (not in code): `fix_supersedes.py` flipped 3
  wrong-direction same-family supersedes edges. The 40 MIL/AR docs + their KG are DB/FAISS state,
  not in git — a fresh checkout starts with an empty corpus.
- Bulk re-ingest is now safe with `KG_AUTO_EXTRACT=True` (serial queue); for a very large batch you
  may still prefer KG-off + a sequential KG pass to keep the task list readable.
- Rapid re-login hits a **429** on `/auth/login` — log in once and only re-auth on 401.
- A few KG supersedes/relation edges carry LLM-classification noise; deterministic blocks derive
  version direction from node names rather than trusting those edges.
