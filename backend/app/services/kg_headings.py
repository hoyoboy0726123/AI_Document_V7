"""Layout/font-based section-heading extraction (doc-type agnostic, no LLM).

Validated on MIL-STD-810H: cleanly extracts METHOD / numbered sections with full
parent hierarchy, while being immune to table cells, TOC pages and figure captions.

Signal is ADAPTIVE per document so it generalises:
  - measure body baseline: median font size + whether body text is mostly non-bold
  - a line is "prominent" if  size >= body_median + 1   OR   (bold AND body mostly non-bold)
Only robust heading SHAPES are kept (others, incl. bold table-cell labels, are dropped):
  - METHOD <n>                       -> kind="method"
  - METHOD <n>, ANNEX <A>            -> kind="annex"  (child of the method)
  - ANNEX <A>                        -> kind="annex"
  - numbered  N(.N){0,3} + real title-> kind="section"   (table cells never match this)

Each heading carries a SCOPED key so numbering that restarts inside an annex
("1. GASOLINE FUELS" under ANNEX A) does NOT collide with the method's own "1. SCOPE":
  method  504.3            -> key "504.3"
  annex   504.3, ANNEX A   -> key "504.3-A"            parent "504.3"
  section 2.2.5.1 in 504.3 -> key "504.3/2.2.5.1"      parent "504.3/2.2.5"
  section 1 inside ANNEX A -> key "504.3-A/1"          parent "504.3-A"

Built on pdfplumber (the same lib the ingest pipeline already uses).
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import pdfplumber

_METHOD_RE = re.compile(r"^\s*METHOD\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_ANNEX_IN_METHOD_RE = re.compile(r"ANNEX\s+([A-Z])\b", re.IGNORECASE)
_ANNEX_RE = re.compile(r"^\s*ANNEX\s+([A-Z])\b", re.IGNORECASE)
_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Za-z].{2,70})$")
_CALLOUT = {"WARNING", "NOTE", "NOTES", "CAUTION", "DANGER", "CONTENTS", "PARAGRAPH",
            "PAGE", "FIGURE", "TABLE", "CONTENTS - CONTINUED"}


@dataclass
class Heading:
    key: str               # unique scoped id (use for canonical_id / part_of)
    number: str            # local display number ("2.2.5.1", "504.3", "A")
    title: str
    kind: str              # method | annex | section
    page: int
    y: float
    parent: Optional[str]  # parent KEY (None → child of the document)


def _local_parent(number: str) -> Optional[str]:
    """2.2.5 -> 2.2 ; 2.2 -> 2 ; 2 -> None (local numeric parent only)."""
    return number.rsplit(".", 1)[0] if "." in number else None


def _page_lines(page):
    """Group a page's words into lines, carrying dominant font size + boldness per line."""
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"], use_text_flow=True)
    except Exception:
        words = page.extract_words(use_text_flow=True)
    lines: Dict[float, list] = {}
    for w in words:
        lines.setdefault(round(float(w.get("top", 0.0)), 1), []).append(w)
    out = []
    for ytop, ws in sorted(lines.items()):
        ws.sort(key=lambda w: w.get("x0", 0.0))
        text = " ".join(w["text"] for w in ws).strip()
        if not text:
            continue
        sizes = [round(float(w.get("size", 0) or 0), 1) for w in ws if w.get("size")]
        dom = Counter(sizes).most_common(1)[0][0] if sizes else 0.0
        bold = sum(1 for w in ws if "bold" in str(w.get("fontname", "")).lower())
        out.append((ytop, dom, bold > len(ws) * 0.6, text))
    return out


def _iter_lines(pdf):
    """Yield (pageno, lines) and gather body baseline stats in one pass."""
    pages, all_sizes, bold_lines, total = [], [], 0, 0
    for pageno, page in enumerate(pdf.pages, start=1):
        lines = _page_lines(page)
        pages.append((pageno, lines))
        for _, dom, is_bold, _t in lines:
            if dom:
                all_sizes.append(dom)
            total += 1
            if is_bold:
                bold_lines += 1
    body = statistics.median([s for s in all_sizes if s]) if any(all_sizes) else 0.0
    bold_is_signal = (bold_lines / max(1, total)) < 0.5
    return pages, body, bold_is_signal


def _classify(txt, pageno, ytop, scope_method, scope_annex):
    """Return (Heading, new_scope_method, new_scope_annex) or (None, ...) if not a heading."""
    mM = _METHOD_RE.match(txt)
    if mM and _ANNEX_IN_METHOD_RE.search(txt):  # "METHOD 504.3, ANNEX A"
        a = _ANNEX_IN_METHOD_RE.search(txt).group(1).upper()
        key = f"{mM.group(1)}-{a}"
        return Heading(key, a, txt, "annex", pageno, ytop, mM.group(1)), mM.group(1), key
    if mM:
        return Heading(mM.group(1), mM.group(1), txt, "method", pageno, ytop, None), mM.group(1), None
    mA = _ANNEX_RE.match(txt)
    if mA:
        a = mA.group(1).upper()
        key = f"{scope_method}-{a}" if scope_method else f"ANNEX-{a}"
        return Heading(key, a, txt, "annex", pageno, ytop, scope_method), scope_method, key
    mN = _NUM_RE.match(txt)
    if mN:
        loc = mN.group(1)
        scope = scope_annex or scope_method
        key = f"{scope}/{loc}" if scope else loc
        lp = _local_parent(loc)
        parent = (f"{scope}/{lp}" if scope else lp) if lp else scope
        return Heading(key, loc, txt, "section", pageno, ytop, parent), scope_method, scope_annex
    return None, scope_method, scope_annex


def _is_heading_line(txt, body, bold_is_signal, dom, is_bold):
    if (body and dom >= body + 1) or (is_bold and bold_is_signal):
        return len(txt) <= 90
    return False


def extract_sections(pdf_bytes: bytes) -> List[Heading]:
    """Return ordered, scope-keyed section headings for the whole document (no LLM)."""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages, body, bold_is_signal = _iter_lines(pdf)
            headings: List[Heading] = []
            sm, sa = None, None  # scope_method, scope_annex
            for pageno, lines in pages:
                if any("CONTENTS" in t.upper() for (_, _, _, t) in lines):
                    continue
                for (ytop, dom, is_bold, txt) in lines:
                    if not _is_heading_line(txt, body, bold_is_signal, dom, is_bold):
                        continue
                    h, sm, sa = _classify(txt, pageno, ytop, sm, sa)
                    if h:
                        headings.append(h)
            return headings
    except Exception:
        return []


def build_section_spec_map(pdf_bytes: bytes) -> Tuple[List[Heading], Dict[str, Set[str]], Set[str]]:
    """One pass: detect headings AND attribute spec ids to the section active at each line.

    Returns (headings, section_specs, doc_specs):
      headings      : List[Heading]   the scope-keyed section tree
      section_specs : dict[key -> set[canonical_id]]  specs cited under each section
      doc_specs     : set[str]        specs appearing before/outside any section
    """
    from . import kg_extractor

    section_specs: Dict[str, Set[str]] = {}
    doc_specs: Set[str] = set()
    headings: List[Heading] = []
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages, body, bold_is_signal = _iter_lines(pdf)
            sm, sa, current = None, None, None  # scopes + active section key
            for pageno, lines in pages:
                if any("CONTENTS" in t.upper() for (_, _, _, t) in lines):
                    continue
                for (ytop, dom, is_bold, txt) in lines:
                    if _is_heading_line(txt, body, bold_is_signal, dom, is_bold):
                        h, sm, sa = _classify(txt, pageno, ytop, sm, sa)
                        if h:
                            headings.append(h)
                            current = h.key
                            continue
                    specs = {s.canonical_id for s in kg_extractor.extract_specs(txt)}
                    if not specs:
                        continue
                    if current:
                        section_specs.setdefault(current, set()).update(specs)
                    else:
                        doc_specs.update(specs)
    except Exception:
        return headings, section_specs, doc_specs
    return headings, section_specs, doc_specs
