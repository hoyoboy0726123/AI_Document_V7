"""Layout/font-based section-heading extraction (doc-type agnostic, no LLM).

Validated on MIL-STD-810H: cleanly extracts METHOD / numbered sections with full
parent hierarchy, while being immune to table cells, TOC pages and figure captions.

Signal is ADAPTIVE per document so it generalises:
  - measure body baseline: median font size + whether body text is mostly non-bold
  - a line is "prominent" if  size >= body_median + 1   OR   (bold AND body mostly non-bold)
Only robust heading SHAPES are kept (others, incl. bold table-cell labels, are dropped):
  - METHOD <n>                       -> kind="method"
  - ANNEX <A>                        -> kind="annex"
  - numbered  N(.N){0,3} + real title-> kind="section"   (table cells never match this)
  - ALL-CAPS standalone title        -> kind="title"

Built on pdfplumber (the same lib the ingest pipeline already uses). Used by the KG
pipeline to create Document->Section(contains) / Section->Section(part_of) /
Section->Standard(references) edges, so individual methods/sections become queryable.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import pdfplumber

_METHOD_RE = re.compile(r"^\s*METHOD\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_ANNEX_RE = re.compile(r"^\s*ANNEX\s+([A-Z])\b", re.IGNORECASE)
_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Za-z].{2,70})$")
_CALLOUT = {"WARNING", "NOTE", "NOTES", "CAUTION", "DANGER", "CONTENTS", "PARAGRAPH",
            "PAGE", "FIGURE", "TABLE", "CONTENTS - CONTINUED"}


@dataclass
class Heading:
    number: str            # "510.7", "4.1.1.2", "A", or "-" for unnumbered titles
    title: str
    kind: str              # method | annex | section | title
    page: int
    y: float               # top y on the page (reading order within a page)
    parent: Optional[str]  # parent number for part_of, or None


def _parent_number(number: str) -> Optional[str]:
    """4.1.1 -> 4.1 ; 4.1 -> 4 ; 4 -> None (methods/annexes have no numeric parent)."""
    if "." in number:
        return number.rsplit(".", 1)[0]
    return None


def _page_lines(page):
    """Group a page's words into lines, carrying dominant font size + boldness per line."""
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
    except Exception:
        words = page.extract_words(use_text_flow=True)
    lines = {}
    for w in words:
        key = round(float(w.get("top", 0.0)), 1)
        lines.setdefault(key, []).append(w)
    out = []
    for ytop, ws in sorted(lines.items()):
        ws.sort(key=lambda w: w.get("x0", 0.0))
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        sizes = [round(float(w.get("size", 0) or 0), 1) for w in ws if w.get("size")]
        dom = Counter(sizes).most_common(1)[0][0] if sizes else 0.0
        bold = sum(1 for w in ws if "bold" in str(w.get("fontname", "")).lower())
        is_bold = bold > len(ws) * 0.6
        out.append((ytop, dom, is_bold, text))
    return out


def extract_sections(pdf_bytes: bytes) -> List[Heading]:
    """Return ordered section headings for the whole document (deterministic, no LLM)."""
    pages_lines = []
    all_sizes, bold_lines, total = [], 0, 0
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                lines = _page_lines(page)
                pages_lines.append((pageno, lines))
                for _, dom, is_bold, _t in lines:
                    if dom:
                        all_sizes.append(dom)
                    total += 1
                    if is_bold:
                        bold_lines += 1
    except Exception:
        return []

    if not pages_lines:
        return []
    body = statistics.median([s for s in all_sizes if s]) if any(all_sizes) else 0.0
    bold_is_signal = (bold_lines / max(1, total)) < 0.5

    def prominent(dom, is_bold):
        return (body and dom >= body + 1) or (is_bold and bold_is_signal)

    headings: List[Heading] = []
    for pageno, lines in pages_lines:
        if any("CONTENTS" in t.upper() for (_, _, _, t) in lines):
            continue  # skip TOC pages
        skip = set()
        for i, (ytop, dom, is_bold, txt) in enumerate(lines):
            if i in skip or not prominent(dom, is_bold) or len(txt) > 90:
                continue
            mM, mA, mN = _METHOD_RE.match(txt), _ANNEX_RE.match(txt), _NUM_RE.match(txt)
            # "METHOD 504.3, ANNEX A ..." → an annex of the method, NOT a fresh method.
            annex_of_method = re.search(r"ANNEX\s+([A-Z])\b", txt, re.IGNORECASE) if mM else None
            if mM and annex_of_method:
                num = f"{mM.group(1)}-{annex_of_method.group(1).upper()}"
                headings.append(Heading(num, txt, "annex", pageno, ytop, mM.group(1)))
            elif mM:
                title = txt
                if i + 1 < len(lines):  # merge METHOD subtitle (next ALL-CAPS line)
                    nxt = lines[i + 1][3].strip()
                    if nxt and nxt.upper() == nxt and not _NUM_RE.match(nxt) and len(nxt) < 60:
                        title = f"{txt} — {nxt}"
                        skip.add(i + 1)
                headings.append(Heading(mM.group(1), title, "method", pageno, ytop, None))
            elif mA:
                headings.append(Heading(mA.group(1), txt, "annex", pageno, ytop, None))
            elif mN:
                num = mN.group(1)
                headings.append(Heading(num, txt, "section", pageno, ytop, _parent_number(num)))
            # NOTE: unnumbered ALL-CAPS "title" lines are intentionally NOT emitted as nodes —
            # they are mostly method subtitles (already merged above) or multi-line WARNING/CAUTION
            # callout text, which would pollute the graph. Only method/annex/numbered sections persist.
    return headings


def build_section_spec_map(pdf_bytes: bytes):
    """One pass: detect section headings AND attribute spec IDs to the section that is
    'active' at each line (line-level, more precise than page-level).

    Returns (headings, section_specs, doc_specs):
      headings      : List[Heading]  (the section tree; method/annex have parent=None)
      section_specs : dict[number -> set[str canonical_id]]  specs cited under each section
      doc_specs     : set[str]  specs that appear before/outside any detected section
    """
    from . import kg_extractor

    headings = extract_sections(pdf_bytes)
    # number -> Heading, only numbered sections/methods/annexes carry a usable key
    by_number = {h.number: h for h in headings if h.number != "-"}
    # quick lookup of heading lines by (page, rounded-y) so the walk can switch section
    head_at = {(h.page, round(h.y, 1)): h.number for h in headings if h.number != "-"}

    section_specs = {}
    doc_specs = set()
    current = None  # active section number

    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                for ytop, dom, is_bold, txt in _page_lines(page):
                    sw = head_at.get((pageno, round(ytop, 1)))
                    if sw is not None:
                        current = sw
                        continue
                    specs = {s.canonical_id for s in kg_extractor.extract_specs(txt)}
                    if not specs:
                        continue
                    if current and current in by_number:
                        section_specs.setdefault(current, set()).update(specs)
                    else:
                        doc_specs.update(specs)
    except Exception:
        pass
    return headings, section_specs, doc_specs
