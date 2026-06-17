"""
Relation classifier — given a chunk of text and a pair of spec entities found
inside it, ask the LLM to label the relation type.

Schema is closed (5 + none) so LLM mistakes can be detected and dropped instead
of polluting the graph with open-vocabulary "relations".
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


REL_TYPES = ("references", "supersedes", "defines", "requires", "derives_from", "none")


@dataclass
class PredictedRelation:
    src: str
    dst: str
    rel_type: str
    confidence: float
    evidence: str


_SYSTEM_PROMPT = (
    "You are a precise classifier for relationships between technical specification "
    "documents. You output ONE JSON object only, no prose, no markdown fences.\n\n"
    "Allowed relation types (output exactly one):\n"
    "- references: source spec mentions/cites the target spec for context or compliance\n"
    "- supersedes: source spec REPLACES the target (target is now obsolete)\n"
    "- defines: source spec DEFINES a term/concept used in the target\n"
    "- requires: source spec MANDATES conformance with the target (stronger than references)\n"
    "- derives_from: source spec is based on / adapted from the target\n"
    "- none: no meaningful relation in this passage\n\n"
    "Schema: {\"rel_type\": str, \"confidence\": float 0-1, \"reason\": str}"
)


def _build_prompt(src: str, dst: str, evidence: str) -> str:
    return (
        f"Passage:\n{evidence}\n\n"
        f"From the passage above, classify the relationship FROM \"{src}\" TO \"{dst}\".\n"
        f"If the passage is the body of \"{src}\" mentioning \"{dst}\", treat \"{src}\" as source.\n"
        "Output JSON only."
    )


def _parse_json_response(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    # Strip fences if model leaked them despite the instruction
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def classify_pair(
    src_canonical: str,
    dst_canonical: str,
    evidence: str,
    *,
    llm_chat,
    model: Optional[str] = None,
    src_doc_canonical: Optional[str] = None,
) -> Optional[PredictedRelation]:
    """
    Classify the relation between two spec entities found in `evidence`.

    `llm_chat` is a callable(messages, model=None, format=None) -> str — pass
    OllamaClient.chat or any LLMProvider.chat. Returns None on parse failure
    or rel_type == "none".
    """
    if src_canonical == dst_canonical:
        return None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(src_canonical, dst_canonical, evidence[:2000])},
    ]

    try:
        raw = llm_chat(messages, model=model, format="json")
    except TypeError:
        # provider may not accept format kwarg
        raw = llm_chat(messages, model=model)
    except Exception as e:
        logger.warning("relation classify LLM call failed: %s", e)
        return None

    parsed = _parse_json_response(raw)
    if not parsed:
        logger.debug("relation classify bad JSON for %s -> %s: %r", src_canonical, dst_canonical, raw[:200])
        return None

    rel_type = str(parsed.get("rel_type", "")).strip().lower()
    if rel_type not in REL_TYPES:
        return None
    if rel_type == "none":
        return None

    try:
        conf = float(parsed.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))

    if conf < 0.3:
        return None

    return PredictedRelation(
        src=src_canonical,
        dst=dst_canonical,
        rel_type=rel_type,
        confidence=conf,
        evidence=evidence[:1000],
    )


def classify_chunk_pairs(
    pairs: List[Tuple[str, str]],
    evidence: str,
    *,
    llm_chat,
    model: Optional[str] = None,
) -> List[PredictedRelation]:
    """Classify a batch of candidate pairs sharing the same evidence chunk."""
    out: List[PredictedRelation] = []
    for src, dst in pairs:
        pred = classify_pair(src, dst, evidence, llm_chat=llm_chat, model=model)
        if pred:
            out.append(pred)
    return out
