"""
通用結構抽取（doc-type 無關，確定性、不呼叫 LLM）。

產生通用 schema 的節點與邊：
  節點： Document / Section（meta.kind 區分 test_item / section …） / 規範實體（沿用 kg_extractor）
  邊：   Document --contains--> Section
         Section  --part_of-->  Section   （依編號階層，如 25.1 ∈ 25）
         Section  --references--> Standard （該章節內出現的規範）
         Document --references--> Standard （不屬於任何章節的規範，fallback）

領域差異一律走 meta.kind（屬性），不新增關係種類，避免文件型別變多後 ontology 爆炸。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from . import kg_extractor

# 編號標題且整行以 "Test" 結尾 → 視為一個測試項目（test_item）。
# 用行尾 $ 錨點避免誤抓含 "test" 字樣的程序步驟（如 "5.Expose ... to the test level"）。
_TESTITEM_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,2})\.?\s*([A-Za-z][A-Za-z0-9 /&\-]{2,70}?[Tt]est)\s*$",
    re.MULTILINE,
)


@dataclass
class SectionNode:
    number: str
    title: str
    kind: str
    page: Optional[int]
    specs: Set[str] = field(default_factory=set)


def _infer_kind(title: str) -> str:
    # 目前只認 test_item；其他文件型別未來在此擴充 kind，節點/關係型別不變。
    return "test_item" if title.strip().lower().endswith("test") else "section"


def _parent_number(number: str) -> Optional[str]:
    """25.1 -> 25 ； 25.1.1 -> 25.1 ； 25 -> None"""
    if "." not in number:
        return None
    return number.rsplit(".", 1)[0]


def extract_structure(chunks) -> Tuple[Dict[str, SectionNode], Set[str]]:
    """
    走訪 chunks（須先依 chunk_index 排序），回傳：
      sections: {number -> SectionNode}（含該章節底下出現的規範 canonical_id）
      doc_level_specs: 不在任何已知章節底下出現的規範（掛在 Document 上）
    """
    sections: Dict[str, SectionNode] = {}
    doc_level_specs: Set[str] = set()
    current: Optional[str] = None

    for ch in chunks:
        text = ch.text or ""
        if not text:
            continue
        # 1) 偵測本 chunk 內的測試項目標題（可能多個）
        for m in _TESTITEM_RE.finditer(text):
            number = m.group(1)
            title = m.group(2).strip()
            if number not in sections:
                sections[number] = SectionNode(
                    number=number,
                    title=title,
                    kind=_infer_kind(title),
                    page=getattr(ch, "page", None),
                )
            current = number
        # 2) 本 chunk 內的規範，歸給「目前所在章節」；無章節則掛文件層
        specs = {s.canonical_id for s in kg_extractor.extract_specs(text)}
        if not specs:
            continue
        if current and current in sections:
            sections[current].specs.update(specs)
        else:
            doc_level_specs.update(specs)

    return sections, doc_level_specs
