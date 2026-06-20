"""
ReAct-style Agent loop.

The LLM emits JSON of one of two shapes per step:
  {"thought": "...", "action": "tool_name", "action_input": {...}}
  {"thought": "...", "final_answer": "..."}

We parse the JSON (strict), execute the tool, append observation, repeat
until final_answer or max_steps. Bad JSON gets one retry with a stricter
instruction; second failure terminates with the raw text as final answer.

Yields a stream of events for SSE consumption.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy.orm import Session

from . import agent_tools, ai
from ..core.config import settings
from .llm_provider import get_llm_provider
from .system_config import SystemConfigService

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_TEMPLATE = """You are a precise document-research agent for a knowledge base of \
technical standards (ISO / IEC / MIL-STD / IEEE / ASTM / JIS / CNS / etc).

You answer the user's question by calling tools step-by-step. Each step you \
output ONE JSON object — no prose, no markdown fences.

Two output shapes:
  {{"thought": "<one-sentence reasoning>", "action": "<tool_name>", "action_input": {{...}}}}
  {{"thought": "<one-sentence reasoning>", "final_answer": "<answer in user's language>"}}

Available tools:
{tools}

Rules:
- Output ONE JSON object per turn, nothing else.
- Stay ANCHORED to the user's original subject. You MAY rephrase a query for better \
retrieval (drop filler words, extract keywords, add synonyms), but you must NOT switch to a \
different topic than what was asked. Retrieved snippets are EVIDENCE, not a redirection — if \
a search surfaces a similarly-named but different item, do not pivot to it. Your final_answer \
must answer the ORIGINAL question's subject.
- Use spec_lookup BEFORE spec_references / spec_supersedes_chain to resolve canonical_id.
- For enumeration / listing questions ("what items/tests does X have", "有哪些", "list all", \
"子項目", "sub-tests of X"), call list_subitems(name) FIRST. It returns the COMPLETE set from \
the knowledge graph; rag_search alone retrieves only top-k chunks and WILL miss items.
- For a question ABOUT a whole category/group ("介紹/說明 X", "tell me about the X tests", \
"these tests", "X 的內容/重點"), or a FOLLOW-UP elaborating on a group, call coverage_check(name) \
to get EVERY sub-item with content. rag_search ranks by keyword/similarity and WILL miss \
sub-items whose pages don't contain the keyword (e.g. a 'Surface Deflection' sub-test under \
'Pressure Test'). If a rag_search for a category looks partial, follow up with coverage_check \
and merge — the final answer must cover ALL sub-items, not only the ones rag_search surfaced.
- If a tool returns no relevant info, try another tool or a different query — do NOT \
hallucinate spec content. If the corpus truly lacks info, say so in final_answer.
- Stop and emit final_answer as soon as you have enough evidence. Brevity is preferred for \
SINGLE-item questions; but for category/group questions completeness wins — make sure every \
sub-item (per list_subitems / coverage_check) is represented before finishing.
- Always answer in the user's language (zh-TW if they wrote Chinese; English otherwise).
- Cite document titles or spec canonical_ids inline when summarizing findings.

Max {max_steps} steps. After that, you MUST emit final_answer."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_step(text: str) -> Optional[Dict[str, Any]]:
    """Return parsed step dict or None if no valid JSON found."""
    if not text:
        return None
    text = text.strip()

    # Try fenced code block first
    fence = _JSON_FENCE_RE.search(text)
    candidates: List[str] = []
    if fence:
        candidates.append(fence.group(1).strip())
    # Then first top-level {...} object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidates.append(m.group(0))

    for raw in candidates:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _build_history_messages(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compress recent QA history into chat messages for context."""
    msgs: List[Dict[str, Any]] = []
    for entry in (conversation_history or [])[-4:]:
        q = (entry.get("question") or "").strip()
        a = (entry.get("answer") or "").strip()
        if q:
            msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a[:1500]})
    return msgs


def _grounded_synthesis(
    db: Session,
    question: str,
    rag_evidence: List[Dict[str, Any]],
    kg_notes: List[str],
    conversation_history: Optional[List[Dict[str, Any]]],
) -> tuple:
    """Phase 0：用已調好的 RAG grounding prompt 重新生成最終答案。

    回傳 (answer, n_rag_used, n_rag_total_unique)，後兩者供 Phase 3 完整度反問使用。

    把 ReAct 過程蒐集到的 rag_search 命中段落（含 title/page）+ KG 關聯整理成
    編號 context，套用「只能依段落、標 [來源]、不可跨來源拼湊」的 RAG 模板再生成一次，
    讓 Agent 答案具備 RAG 等級的引用與防幻覺紀律。總長度受 num_ctx 預算限制。
    """
    budget = ai.effective_rag_budget()

    # 先去重，算出「實際有幾筆不重複的檢索證據」(total_unique)，供 Phase 3 完整度判斷
    deduped: List[tuple] = []
    seen: set = set()
    for ev in rag_evidence:
        text = (ev.get("snippet") or ev.get("text") or "").strip()
        if not text:
            continue
        key = (ev.get("document_id"), ev.get("page"), text[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((ev, text))
    total_unique = len(deduped)

    # 在預算內塞入 context；裝不下的就是「檢索到但未展開」的來源
    contexts: List[Dict[str, Any]] = []
    used_sources: List[Dict[str, Any]] = []  # 供前端可點/可預覽的結構化來源（對齊 RAG）
    used = 0
    for ev, text in deduped:
        if used + len(text) > budget:
            text = text[: max(0, budget - used)]
        if not text:
            break
        contexts.append({
            "source_num": len(contexts) + 1,
            "title": ev.get("title") or "",
            "page": ev.get("page"),
            "page_gap": None,
            "text": text,
        })
        used_sources.append({
            "document_id": ev.get("document_id"),
            "title": ev.get("title") or "",
            "page": ev.get("page"),
            "snippet": ev.get("snippet") or text,
            "score": ev.get("score"),
        })
        used += len(text)
        if used >= budget:
            break
    n_rag_used = len(contexts)

    # 把 KG / spec 關聯當作額外一個來源塊（若還有預算）
    if kg_notes and used < budget:
        note_text = ("關聯資訊（知識圖譜）:\n" + "\n".join(kg_notes))[: max(0, budget - used)]
        if note_text.strip():
            contexts.append({
                "source_num": len(contexts) + 1,
                "title": "知識圖譜關聯",
                "page": None,
                "page_gap": None,
                "text": note_text,
            })

    if not contexts:
        return None, [], 0, total_unique

    prompts = SystemConfigService(db).get_rag_prompts()
    answer = ai.generate_rag_answer(
        question,
        contexts,
        conversation_history=conversation_history,
        system_prompt=prompts["system_prompt"],
        user_template=prompts["user_template"],
    )
    return answer, used_sources, n_rag_used, total_unique


def _coverage_note(n_unused_sources: int, kg_edges_seen: int) -> str:
    """Phase 3：依啟發式產生「還有更多、要不要深掘」的反問（不額外呼叫 LLM）。

    只在「確實還有未展開的檢索來源或圖譜關聯」時才附加，避免每次都問。
    """
    bits: List[str] = []
    if n_unused_sources > 0:
        bits.append(f"另有約 {n_unused_sources} 段相關內容因長度限制未在本次展開")
    if kg_edges_seen > 0:
        bits.append(f"知識圖譜中已找到 {kg_edges_seen} 條規範關聯，可再往上追引用鏈／版本鏈")
    if not bits:
        return ""
    return "\n\n---\n📌 " + "；".join(bits) + "。需要我針對哪一項深入查嗎？"


_ENUM_RE = re.compile(
    r"(有哪些|哪些|列出|列舉|清單|子項目|有什麼|包含哪些|底下有|下有|"
    r"sub[- ]?items?|list all|list the|what .*(items|tests|sub)|which .*(items|tests))",
    re.IGNORECASE,
)


def _looks_like_enumeration(q: str) -> bool:
    """判斷是否為列舉題（有哪些/列出/子項目…），用於確定性 KG 列舉 fallback。"""
    return bool(_ENUM_RE.search(q or ""))


_OVERVIEW_RE = re.compile(
    r"(介紹|說明|概覽|概述|描述|重點|摘要|整理|內容|overview|brief|describe|summar|"
    r"tell me about|rundown|walk me through|各[項個種子]|每[項個種]|each\b|"
    r"all (the )?(tests|items|sub))",
    re.IGNORECASE,
)


def _looks_like_overview(q: str) -> bool:
    """判斷是否為『類別/概覽』題（介紹/說明/各項/each…）。實際是否接管仍需 KG 命中有子項的父節點。"""
    return bool(_OVERVIEW_RE.search(q or ""))


_ASPECT_RE = [
    ("criteria", re.compile(r"(判定標準|判定基準|驗收標準|合格標準|測試標準|criteria|criterion)", re.IGNORECASE)),
    ("specification", re.compile(r"(測試規格|規格|測試條件|specification|\bspec\b)", re.IGNORECASE)),
    ("objective", re.compile(r"(測試目的|目的|用途|objective)", re.IGNORECASE)),
]


def _detect_aspect(q: str) -> Optional[str]:
    """偵測『細節橫跨多項目』問題的面向：判定標準 / 規格 / 目的。"""
    for aspect, rx in _ASPECT_RE:
        if rx.search(q or ""):
            return aspect
    return None


def _group_summary(db: Session, matched: str, conversation_history) -> str:
    """對一組測試做 2~4 句摘要（自行檢索內容），供列舉題附加說明。"""
    try:
        rs = agent_tools.run_tool(db, "rag_search", {"query": matched, "top_k": 5})
        ev = rs.get("results", []) if isinstance(rs, dict) else []
        if not ev:
            return ""
        sq = (
            f"請用 2~4 句話摘要說明「{matched}」這組測試整體在測試什麼、目的為何，"
            "以段落呈現；不要再列出子項目清單。"
        )
        ans, _s, _u, _t = _grounded_synthesis(db, sq, ev, [], conversation_history)
        return ans.strip() if ans else ""
    except Exception as e:
        logger.warning("group summary failed: %s", e)
        return ""


_STRUCT_SNIPPET_CAP = 1400  # 每個子項餵進合成的內容上限（夠長到容納像 12.1 那種多段規格）


def _cap_clean(text: str, cap: int) -> str:
    """截斷到 cap 字內，但切在換行/句界，避免把內容切成半字（如「each 3 c」）。"""
    if not text or len(text) <= cap:
        return text
    cut = text[:cap]
    for sep in ("\n", "。", "; ", "；", ". "):
        i = cut.rfind(sep)
        if i > cap * 0.6:
            return cut[: i + len(sep)].rstrip()
    return cut.rstrip()


def _structural_evidence(db: Session, question: str, conversation_history) -> Optional[Dict[str, Any]]:
    """結構性問題（判定標準/規格/目的/列舉/概覽）→ 用 KG 把「全部子項的內容」撈齊，
    回傳 {rag_evidence, kg_notes, matched} 供 `_grounded_synthesis`(LLM) 合成最完整答案。

    重點:KG 在這裡的角色是「保證證據完整」(含關鍵字撈不到的子項),最終答案仍由 LLM 結合
    RAG 風格的引用紀律合成 —— 不再用套版直接回傳。對象名稱從『問題 + 最近一輪歷史』解析。
    """
    aspect = _detect_aspect(question)
    is_overview = _looks_like_overview(question)
    is_enum = _looks_like_enumeration(question)
    if not (aspect or is_overview or is_enum):
        return None

    search_text = question or ""
    if conversation_history:
        last = conversation_history[-1]
        search_text = f"{question} {last.get('question', '')} {(last.get('answer', '') or '')[:200]}"

    # 依問題面向取內容：有 aspect → 抓該面向段落；否則抓整節內容。
    if aspect:
        det = agent_tools.run_tool(db, "get_subitem_details", {"name": search_text, "aspect": aspect})
        items = det.get("items") or []
        content_key = "detail"
    else:
        det = agent_tools.run_tool(db, "coverage_check", {"name": search_text})
        items = det.get("items") or []
        content_key = "excerpt"

    matched = det.get("matched")
    if not matched or not items:
        return None

    rag_evidence: List[Dict[str, Any]] = []
    all_names: List[str] = []
    for it in items:
        num, nm = it.get("number"), it.get("name")
        label = f"{(str(num) + ' ') if num else ''}{nm}".strip()
        if nm:
            all_names.append(label)
        content = (it.get(content_key) or "").strip()
        if not content or not it.get("document_id"):
            continue
        rag_evidence.append({
            "document_id": it["document_id"],
            "title": label or nm,
            "page": it.get("page"),
            "snippet": _cap_clean(content, _STRUCT_SNIPPET_CAP),
            "score": None,
        })

    if not rag_evidence:
        return None

    # 權威完整清單當作 KG note，明確要求 LLM「全部涵蓋」(即使某些子項內容被預算截短也不可遺漏)。
    kg_notes: List[str] = []
    if all_names:
        kg_notes.append(
            f"「{matched}」在知識圖譜結構下共有 {len(all_names)} 個子項目，"
            f"回答必須完整涵蓋每一個：{'、'.join(all_names)}"
        )
    refs = det.get("references") or []
    if refs:
        kg_notes.append(f"「{matched}」引用的標準：{'、'.join(refs)}")

    return {"rag_evidence": rag_evidence, "kg_notes": kg_notes, "matched": matched}


def _seed_evidence_via_rag(db: Session, question: str, top_k: int = 5) -> tuple:
    """用「使用者原始問題」跑與 /rag/query 完全相同的檢索，回傳 (evidence, confidence)。

    走 services.retrieval（檢索邏輯的唯一實作），讓 Agent 的基準證據與 RAG 模式逐字一致
    （已驗證對地端與雲端都能撈到正確的表格段）。
    confidence = 最相關段落的 cross-encoder 分數（None 表示 CE 不可用）。
    """
    from . import rerank, retrieval

    embeddings = ai.embed_texts([question])
    if not embeddings:
        return [], None
    filtered = retrieval.hybrid_retrieve(db, question, embeddings[0], top_k)
    if not filtered:
        return [], None
    confidence = rerank.top_relevance(question, [c for c, _ in filtered])
    keep_ids = retrieval.context_keep_ids(filtered)
    evidence: List[Dict[str, Any]] = []
    ctx_used = 0
    for chunk, score in filtered:
        if chunk.id not in keep_ids:
            continue
        text = retrieval.context_text_budgeted(db, chunk, ctx_used)
        ctx_used += len(text)
        evidence.append({
            "document_id": chunk.document_id,
            "title": chunk.document.title if chunk.document else "",
            "page": chunk.page,
            "score": score,
            "snippet": text,
        })
    return evidence, confidence


def _build_enumeration_answer(db: Session, chosen: Dict[str, Any], rag_evidence, conversation_history):
    """用 KG 的權威完整子項清單組裝「確定性列舉答案」(body, sources)，不經會漏項的 grounded 合成。

    grounded 合成受 num_ctx 預算限制，6 個子項會被截到只展開前 2 個；列舉題要的是「完整列出」，
    所以直接照 KG 結構列全部，再附一段整體摘要說明（摘要才用合成）。
    """
    lines = [f"「{chosen['matched']}」共有 {len(chosen['subitems'])} 個子項目："]
    enum_sources: List[Dict[str, Any]] = []
    for it in chosen["subitems"]:
        num = it.get("number")
        lines.append(f"- {(str(num) + ' ') if num else ''}{it.get('name', '')}")
        if it.get("document_id"):
            enum_sources.append({
                "document_id": it.get("document_id"),
                "title": it.get("name", ""),
                "page": it.get("page"),
                "snippet": f"{(str(num) + ' ') if num else ''}{it.get('name', '')}",
                "score": None,
            })
    if chosen.get("references"):
        lines.append(f"\n引用標準：{'、'.join(chosen['references'])}")
    lines.append("\n（以上為知識圖譜結構的完整列舉）")
    body = "\n".join(lines)

    # 不只列清單：再用檢索內容對這組測試做摘要說明（這段才用合成，漏不漏項不影響清單完整性）。
    evidence_for_summary = list(rag_evidence or [])
    if not evidence_for_summary:
        try:
            rs = agent_tools.run_tool(db, "rag_search", {"query": chosen["matched"], "top_k": 5})
            if isinstance(rs, dict):
                evidence_for_summary = rs.get("results", []) or []
        except Exception:
            evidence_for_summary = []
    if evidence_for_summary:
        try:
            sq = (
                f"請用 2~4 句話摘要說明「{chosen['matched']}」這組測試整體在測試什麼、目的為何，"
                "以段落呈現；不要再列出子項目清單。"
            )
            s_ans, _s, _u, _t = _grounded_synthesis(db, sq, evidence_for_summary, [], conversation_history)
            if s_ans and s_ans.strip():
                body += f"\n\n📝 摘要說明：\n{s_ans.strip()}"
        except Exception as e:
            logger.warning("enumeration summary failed: %s", e)
    return body, enum_sources


def run_agent(
    db: Session,
    question: str,
    *,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 8,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield events:
      {"type": "thought", "step": n, "text": ...}
      {"type": "tool_call", "step": n, "tool": str, "input": dict}
      {"type": "observation", "step": n, "tool": str, "output": dict}
      {"type": "final", "text": str}
      {"type": "error", "message": str}
    """
    # 純列舉題（「有哪些子項目 / 列出全部」且非規格/判定/目的等細節面向）→ 直接用 KG 確定性完整列舉。
    # 不走下方 _structural_evidence 的 grounded 合成：合成受 num_ctx 預算限制會漏項（6 子項只展開 2 個）。
    if _looks_like_enumeration(question) and _detect_aspect(question) is None:
        try:
            fb = agent_tools.run_tool(db, "list_subitems", {"name": question})
        except Exception as e:
            logger.warning("enumeration list_subitems failed: %s", e)
            fb = None
        if isinstance(fb, dict) and fb.get("subitems"):
            yield {"type": "thought", "step": 0,
                   "text": f"偵測到列舉題：用知識圖譜完整列出「{fb.get('matched')}」全部子項目。"}
            body, esrc = _build_enumeration_answer(
                db,
                {"matched": fb.get("matched"), "subitems": fb.get("subitems"),
                 "references": fb.get("references") or []},
                None, conversation_history,
            )
            yield {"type": "final", "text": body, "sources": esrc}
            return

    # 結構性問題（逐項細節 / 概覽）：用 KG 把「全部子項內容」撈齊(保證完整、含關鍵字漏撈的子項)，
    # 再交給 RAG grounded 合成由 LLM 產生最完整、有引用的答案 —— 不用套版直接回傳。
    try:
        struct = _structural_evidence(db, question, conversation_history)
    except Exception as e:
        logger.warning("structural evidence gather failed: %s", e)
        struct = None
    if struct:
        yield {"type": "thought", "step": 0,
               "text": f"偵測到結構性問題：用知識圖譜撈齊「{struct.get('matched')}」全部子項，再結合 RAG 合成完整答案。"}
        try:
            ans, sources, _u, _t = _grounded_synthesis(
                db, question, struct["rag_evidence"], struct["kg_notes"], conversation_history
            )
        except Exception as e:
            logger.warning("structural grounded synthesis failed: %s", e)
            ans, sources = None, []
        if ans and ans.strip():
            yield {"type": "final", "text": ans.strip(), "sources": sources or []}
            return
        # 合成失敗 → 落回下方完整 ReAct loop

    llm = get_llm_provider()
    tools_desc = agent_tools.render_tools_for_prompt()
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools=tools_desc, max_steps=max_steps)

    scratchpad: List[Dict[str, Any]] = []  # alternating assistant (tool call) / user (observation)
    history_messages = _build_history_messages(conversation_history or [])

    final_text: Optional[str] = None
    # Phase 0：蒐集證據供最後 grounded 合成（rag_search 命中段落 + KG/spec 關聯）
    rag_evidence: List[Dict[str, Any]] = []
    kg_notes: List[str] = []
    kg_edges_seen = 0  # Phase 3：圖譜關聯數，供完整度反問
    structural_results: List[Dict[str, Any]] = []  # list_subitems 的權威完整清單（列舉題確定性作答）
    seeded: List[Dict[str, Any]] = []   # 保留供後段「補過仍無證據」的低信心兜底
    seed_conf: Optional[float] = None
    threshold = getattr(settings, "RAG_LOWCONF_CE_THRESHOLD", 0.15)

    # 雲地相容 seed：先用「使用者原始問題」做一次確定性檢索，把命中段落放進 rag_evidence。
    # 走 /rag/query 完全相同的檢索，保證最終 grounded 合成握有「對的證據」。
    # 注意：此處「不」因低信心就短路結束——改由後段「合成 → 不足則自動補充一輪 → 仍不足才低信心」
    # 處理，避免在 ReAct 工具迴圈跑之前就放棄 KG/結構工具能回答的題型（如版本/取代/引用類）。
    try:
        seeded, seed_conf = _seed_evidence_via_rag(db, question, top_k=5)
        rag_evidence.extend(seeded)
        if seeded:
            yield {"type": "thought", "step": 0,
                   "text": "先以原始問題做一次基準檢索（與 RAG 同一條 pipeline），確保證據齊全且與所用模型無關。"}
    except Exception as e:
        logger.warning("agent seed retrieval failed: %s", e)

    for step in range(1, max_steps + 1):
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": question},
        ]
        messages.extend(scratchpad)

        # Strong nudge if we're approaching the cap
        if step == max_steps:
            messages.append({
                "role": "user",
                "content": "You are at the final step. Emit final_answer JSON now.",
            })

        try:
            raw = llm.chat(messages, format="json")
        except TypeError:
            raw = llm.chat(messages)
        except Exception as e:
            logger.error("agent llm chat failed at step %d: %s", step, e)
            yield {"type": "error", "message": f"LLM 呼叫失敗：{e}"}
            return

        parsed = _parse_step(raw)
        if parsed is None:
            # one retry with stricter wording
            retry_msgs = messages + [{
                "role": "user",
                "content": (
                    "Your previous output was not valid JSON. Output a single JSON "
                    "object now matching either {thought, action, action_input} or "
                    "{thought, final_answer}. No prose, no fences."
                ),
            }]
            try:
                raw2 = llm.chat(retry_msgs, format="json")
            except TypeError:
                raw2 = llm.chat(retry_msgs)
            except Exception as e:
                yield {"type": "error", "message": f"LLM 重試失敗：{e}"}
                return
            parsed = _parse_step(raw2)
            if parsed is None:
                logger.warning("agent step %d gave no parseable JSON; using raw text", step)
                final_text = (raw or raw2 or "").strip() or "暫無足夠資訊"
                break

        thought = str(parsed.get("thought") or "").strip()
        if thought:
            yield {"type": "thought", "step": step, "text": thought}

        if "final_answer" in parsed:
            final_text = str(parsed.get("final_answer") or "").strip()
            break

        action = str(parsed.get("action") or "").strip()
        # 容錯：LLM 偶爾把工具名叫錯（get_sumerules_item_details…）→ 正規化成有效名，
        # 讓 SSE 事件、run_tool 分派、以及下方依 action 名的後處理都用同一個正確名稱。
        if action and action not in agent_tools.TOOLS:
            action = agent_tools.resolve_tool_name(action) or action
        action_input = parsed.get("action_input") or {}
        if not isinstance(action_input, dict):
            action_input = {"raw": str(action_input)}

        if not action:
            # missing action AND no final_answer — terminate with whatever we have
            final_text = thought or "暫無足夠資訊"
            break

        yield {"type": "tool_call", "step": step, "tool": action, "input": action_input}

        observation = agent_tools.run_tool(db, action, action_input)
        yield {"type": "observation", "step": step, "tool": action, "output": observation}

        # Phase 0：累積證據供最後 grounded 合成
        if isinstance(observation, dict):
            if action == "rag_search":
                for res in observation.get("results", []):
                    if isinstance(res, dict):
                        rag_evidence.append(res)
            elif action == "list_subitems" and "error" not in observation:
                # 把「完整子項目清單」當成權威證據餵進合成，避免合成只用 RAG 片段而漏項。
                items = observation.get("subitems") or []
                names = "、".join(i.get("name", "") for i in items if i.get("name"))
                cnt = observation.get("subitem_count", len(items))
                if items:
                    structural_results.append({
                        "matched": observation.get("matched"),
                        "subitems": items,
                        "references": observation.get("references") or [],
                    })
                if names:
                    kg_notes.append(f"「{observation.get('matched')}」的完整子項目（共 {cnt} 項，來自知識圖譜結構）：{names}")
                refs = observation.get("references") or []
                if refs:
                    kg_notes.append(f"「{observation.get('matched')}」引用的標準：{'、'.join(refs)}")
            elif action in ("spec_references", "spec_supersedes_chain", "spec_lookup", "document_get"):
                if "error" not in observation:
                    kg_notes.append(f"{action} → " + json.dumps(observation, ensure_ascii=False)[:900])
                    if action == "spec_references":
                        kg_edges_seen += len(observation.get("outgoing", [])) + len(observation.get("incoming", []))

        # Append to scratchpad so the LLM sees its previous tool call + result
        scratchpad.append({
            "role": "assistant",
            "content": json.dumps(parsed, ensure_ascii=False),
        })
        scratchpad.append({
            "role": "user",
            "content": (
                f"Observation from {action}:\n"
                + json.dumps(observation, ensure_ascii=False)[:6000]
            ),
        })

    if final_text is None:
        final_text = "已達最大步數，未能整合出最終答案。建議縮小問題範圍後重試。"

    # 列舉題：用 list_subitems 的權威完整清單作答（不經會漏項的 RAG 合成）。
    ql = question.lower()
    chosen = None
    for sr in structural_results:
        if (sr.get("matched") or "").lower() in ql and sr.get("subitems"):
            chosen = sr
            break
    if chosen is None and len(structural_results) == 1 and structural_results[0].get("subitems"):
        chosen = structural_results[0]
    # 確定性 fallback：即使這一輪 LLM 沒呼叫 list_subitems，列舉題仍直接從問題字串解析實體並完整列舉。
    if chosen is None and _looks_like_enumeration(question):
        fb = agent_tools.run_tool(db, "list_subitems", {"name": question})
        if isinstance(fb, dict) and fb.get("subitems"):
            chosen = {
                "matched": fb.get("matched"),
                "subitems": fb.get("subitems"),
                "references": fb.get("references") or [],
            }

    if chosen:
        body, enum_sources = _build_enumeration_answer(db, chosen, rag_evidence, conversation_history)
        yield {"type": "final", "text": body, "sources": enum_sources}
        return

    # Phase 0/3：合成 → 充足性檢查 → 不足則自動補充一輪 → 重新合成 → 仍不足才低信心兜底。
    final_sources: List[Dict[str, Any]] = []

    def _synthesize():
        """用目前的 rag_evidence + kg_notes 跑帶引用的合成，回傳 (答案含覆蓋反問, 來源, n_used)。"""
        if not (rag_evidence or kg_notes):
            return None, [], 0
        try:
            g, srcs, n_used, n_total = _grounded_synthesis(
                db, question, rag_evidence, kg_notes, conversation_history
            )
            if g and g.strip():
                note = _coverage_note(max(0, n_total - n_used), kg_edges_seen)
                return g.strip() + (note or ""), srcs, n_used
        except Exception as e:
            logger.warning("grounded synthesis failed: %s", e)
        return None, [], 0

    synth, syn_sources, n_used = _synthesize()

    # 充足性檢查（確定性，不靠 LLM）：合成不出，或「迴圈完全沒撈到 KG 關聯且 seed 向量信心偏低」
    # → 答案只靠弱向量證據，視為不足，自動補充一輪（更廣檢索 + 規範關聯展開）後重新合成。
    weak = (synth is None) or (not kg_notes and seed_conf is not None and seed_conf < threshold)
    if weak:
        try:
            yield {"type": "thought", "step": max_steps + 1,
                   "text": "初次證據不足，自動補充一輪（更廣檢索＋規範關聯展開）後重新作答。"}
            # (1) 更廣的檢索（提高 top_k），去重併入既有證據
            more, _ = _seed_evidence_via_rag(db, question, top_k=10)
            seen = {(e.get("document_id"), e.get("page"),
                     (e.get("snippet") or e.get("text") or "")[:48]) for e in rag_evidence}
            for e in more:
                k = (e.get("document_id"), e.get("page"),
                     (e.get("snippet") or e.get("text") or "")[:48])
                if k not in seen:
                    rag_evidence.append(e)
                    seen.add(k)
            # (2) 問題若含規範 ID → 自動 KG 展開引用關聯（「向量弱但 KG 強」題型的關鍵補充）
            from . import kg_extractor
            for sm in kg_extractor.extract_specs(question)[:4]:
                obs = agent_tools.run_tool(db, "spec_references", {"spec_id": sm.canonical_id, "hops": 1})
                if isinstance(obs, dict) and "error" not in obs:
                    kg_notes.append("spec_references → " + json.dumps(obs, ensure_ascii=False)[:900])
                    kg_edges_seen += len(obs.get("outgoing", [])) + len(obs.get("incoming", []))
            synth, syn_sources, n_used = _synthesize()
        except Exception as e:
            logger.warning("agent supplement round failed: %s", e)

    if synth and synth.strip():
        final_text = synth
        final_sources = syn_sources
    elif not (rag_evidence or kg_notes):
        # 補過仍無任何證據 → 誠實低信心兜底（真正的最後手段，移到流程末端）。
        closest = [{"title": ev.get("title"), "page": ev.get("page"), "text": ev.get("snippet")}
                   for ev in seeded[:3]]
        if closest:
            final_text = ai.low_confidence_answer(question, closest)
            final_sources = [{"document_id": ev.get("document_id"), "title": ev.get("title"),
                              "page": ev.get("page"), "snippet": ev.get("snippet"),
                              "score": ev.get("score")} for ev in seeded[:5]]
    # else：有證據但合成不出 → 保留 ReAct 迴圈自己的 final_text。

    yield {"type": "final", "text": final_text, "sources": final_sources}
