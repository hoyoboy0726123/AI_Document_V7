r"""
TEMPORARY verification harness (delete after the experiment).

Goal: settle whether the *whole wide table* reproduction failure
(PC-only-mode radiated-emission, 11 columns, from the EMI doc) is a
gemma-12b model-capability limit or something in our pipeline.

It does NOT reinvent anything — it exercises the REAL wired path:
  * same retrieval / context-budget / message-build as /rag/query,
  * embeddings stay on Ollama (the FAISS index was built with them),
  * only the synthesis LLM is flipped to Gemini by setting LLM_PROVIDER
    at runtime, then calling ai.generate_rag_answer() unchanged.

If Gemini reproduces the full table from the same context that gemma
mangles, the bottleneck is the local model, not retrieval.

Run from backend/ with the venv python, after exporting the key:
    set GEMINI_API_KEY=...            (cmd)
    $env:GEMINI_API_KEY="..."         (PowerShell)
    .venv\Scripts\python.exe _gemini_probe.py

Optional env:
    GEMINI_PROBE_MODEL   (default: gemini-2.5-flash)
    GEMINI_PROBE_TOPK    (default: 4)

The key is read from the environment only — never hard-coded, logged, or committed.
"""
from __future__ import annotations

import os
import types

from app.core.config import settings
from app.database import SessionLocal
from app.services import ai
from app.services import llm_provider
from app.services.system_config import SystemConfigService
from app.api.v1 import rag as rag_api


def _fake_payload(top_k: int):
    """Minimal stand-in for schemas.RAGQueryRequest — only the attrs
    _hybrid_filtered / query_rag read."""
    return types.SimpleNamespace(
        question="", top_k=top_k, document_id=None, classification_id=None,
        project_id=None, folder_ids=None, conversation_history=None,
        skip_ai_understanding=True, use_ai_fallback=False,
    )


def build_context(db, question: str, top_k: int):
    """Replicate /rag/query retrieval to produce the SAME context blocks."""
    config_service = SystemConfigService(db)
    vector_config = config_service.get_vector_config()
    embeddings = ai.embed_texts([question])  # stays on Ollama → index-compatible
    if not embeddings:
        raise RuntimeError("embedding failed (Ollama up?)")

    payload = _fake_payload(top_k)
    filtered = rag_api._hybrid_filtered(db, payload, question, embeddings[0], vector_config, top_k)
    if not filtered:
        return [], []

    primary_page = filtered[0][0].page or 0
    keep_ids = rag_api._context_keep_ids(filtered)
    contexts, src_pages, ctx_used = [], [], 0
    for chunk, score in filtered:
        src_pages.append(chunk.page or 0)
        if chunk.id not in keep_ids:
            continue
        text = rag_api._context_text_budgeted(db, chunk, ctx_used)
        ctx_used += len(text)
        contexts.append({
            "source_num": len(contexts) + 1,
            "title": chunk.document.title,
            "page": chunk.page or 0,
            "page_gap": abs((chunk.page or 0) - primary_page) if primary_page and chunk.page else None,
            "text": text,
        })
    return contexts, src_pages


def main():
    model = os.environ.get("GEMINI_PROBE_MODEL", "gemini-2.5-flash")
    top_k = int(os.environ.get("GEMINI_PROBE_TOPK", "4"))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY not set in environment")

    # Flip synthesis to Gemini for THIS process only (does not touch DB/.env).
    settings.GEMINI_API_KEY = key
    settings.LLM_PROVIDER = "gemini"
    settings.LLM_MODEL = model
    llm_provider.invalidate()  # force the factory to rebuild as Gemini
    provider = llm_provider.get_llm_provider()
    print(f"active LLM provider = {provider.name} (model={model})", flush=True)

    questions = [
        ("WHOLE TABLE",
         "請完整列出 PC only mode 輻射發射測試 (30MHz~1GHz) 的整張數據表格，"
         "每一列（每個頻率點）都要，以 markdown 表格呈現，不要省略任何列。"),
        ("CELL 101.78",
         "PC only mode 輻射發射測試中，頻率 101.78 MHz 那一列的 Result (dBuV) 和 Margin (dB) 各是多少？"),
        ("CELL 887.48",
         "PC only mode 輻射發射測試中，頻率 887.48 MHz 那一列的 Result (dBuV) 和 Limit (dBuV) 各是多少？"),
    ]

    db = SessionLocal()
    try:
        rag_prompts = SystemConfigService(db).get_rag_prompts()
        for label, q in questions:
            print("=" * 70, flush=True)
            print(f"[{label}] {q}", flush=True)
            contexts, src_pages = build_context(db, q, top_k)
            print(f"  sources(pages)={src_pages}  blocks_in_LLM={len(contexts)}", flush=True)
            if not contexts:
                print("  -> no context, skipping", flush=True)
                continue
            try:
                ans = ai.generate_rag_answer(
                    q, contexts, None,
                    system_prompt=rag_prompts["system_prompt"],
                    user_template=rag_prompts["user_template"],
                )
            except Exception as e:
                ans = f"<gemini error: {e}>"
            print(f"\n  --- GEMINI ({model}) ---\n{ans}\n", flush=True)
    finally:
        db.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
