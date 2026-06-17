"""
Lightweight ONNX OCR backend: RapidLayout + RapidTable + RapidOCR.

Why this exists: PP-StructureV3 (paddle) is heavy and has Windows-CPU native
segfault risks. This backend is pure onnxruntime (no torch/paddle), ~4s/page on
CPU with mobile models, and produces the SAME OCRBlock schema as the
PP-StructureV3 path so the two are interchangeable at the block level.

Pipeline per page:
  render(dpi/max_dim) -> RapidLayout(PP-DocLayoutV3) detect regions
    -> table region : crop -> RapidTable -> HTML -> markdown pipe-table
    -> text  region : crop -> RapidOCR   -> joined text

Tier / version / device are config-driven and validated live:
  mobile + PP-OCRv4  ~4s/page, clean structure, occasional minor slip
  server + PP-OCRv5  near-perfect, but ~87s/page on CPU (fast on GPU)

Engines are cached singletons keyed by (tier, version, device); call
`invalidate()` after admin changes OCR settings.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .pdf_image import pdf_page_to_image, get_pdf_page_count
from .ocr_pipeline import OCRBlock, normalize_ocr_blocks, _html_table_to_markdown
from ..core.config import settings

logger = logging.getLogger(__name__)

# Layout labels we treat as table vs. drop entirely; everything else with text
# is OCR'd as a text block.
_TABLE_LABELS = {"table"}
_SKIP_LABELS = {
    "chart", "figure", "image", "seal", "figure_title", "formula_number",
    "number", "footer_image", "header_image", "vision_footnote", "stamp",
    "display_formula", "inline_formula",
}

_lock = threading.Lock()
_engines: Optional[Dict[str, Any]] = None
_engine_sig: Optional[Tuple[str, str, str]] = None


def _resolve_cfg() -> Tuple[str, str, str]:
    tier = (getattr(settings, "OCR_MODEL_TIER", "mobile") or "mobile").lower()
    version = getattr(settings, "OCR_VERSION", "PP-OCRv4") or "PP-OCRv4"
    device = (getattr(settings, "OCR_DEVICE", "cpu") or "cpu").lower()
    return tier, version, device


def _ocr_params(tier: str, version: str, device: str) -> Dict[str, Any]:
    """Build the rapidocr params dict (values MUST be Enum instances)."""
    from rapidocr import ModelType, OCRVersion

    mt = ModelType.SERVER if tier == "server" else ModelType.MOBILE
    ver = OCRVersion.PPOCRV5 if version == "PP-OCRv5" else OCRVersion.PPOCRV4
    params: Dict[str, Any] = {
        "Det.model_type": mt, "Det.ocr_version": ver,
        "Rec.model_type": mt, "Rec.ocr_version": ver,
    }
    if device == "gpu":
        # Requires onnxruntime-gpu on the host (e.g. the 5090 box). Best-effort:
        # the engine ignores/falls back to CPU if CUDA EP is unavailable.
        params["EngineConfig.onnxruntime.use_cuda"] = True
    return params


def _build_engines(tier: str, version: str, device: str) -> Dict[str, Any]:
    from rapid_layout import RapidLayout, RapidLayoutInput, ModelType as LMT
    from rapid_table import RapidTable, RapidTableInput, ModelType as TMT
    from rapidocr import RapidOCR

    params = _ocr_params(tier, version, device)
    logger.info("rapid_ocr: building engines tier=%s version=%s device=%s", tier, version, device)
    layout = RapidLayout(RapidLayoutInput(model_type=LMT.PP_DOC_LAYOUTV3))
    table = RapidTable(RapidTableInput(model_type=TMT.PPSTRUCTURE_EN, use_ocr=True, ocr_params=params))
    text_ocr = RapidOCR(params=params)
    return {"layout": layout, "table": table, "text_ocr": text_ocr}


def _get_engines() -> Dict[str, Any]:
    global _engines, _engine_sig
    sig = _resolve_cfg()
    with _lock:
        if _engines is None or sig != _engine_sig:
            _engines = _build_engines(*sig)
            _engine_sig = sig
        return _engines


def invalidate() -> None:
    """Drop cached engines so the next call rebuilds with current settings."""
    global _engines, _engine_sig
    with _lock:
        _engines = None
        _engine_sig = None


def _html_of(out: Any) -> str:
    for attr in ("pred_html", "pred_htmls"):
        v = getattr(out, attr, None)
        if v:
            return v[0] if isinstance(v, (list, tuple)) else v
    return ""


def _reading_order(boxes, names, scores):
    """Sort regions roughly top-to-bottom, left-to-right (row-banded)."""
    items = list(zip(boxes, names, scores))
    # band y by 24px so same-row regions stay left-to-right
    items.sort(key=lambda it: (round(float(it[0][1]) / 24.0), float(it[0][0])))
    return items


def _ocr_text(text_ocr, crop: Image.Image) -> str:
    try:
        res = text_ocr(np.asarray(crop))
    except Exception as exc:
        logger.debug("rapid_ocr text region failed: %s", exc)
        return ""
    txts = getattr(res, "txts", None)
    if not txts:
        return ""
    return " ".join(t for t in txts if t).strip()


def rapid_page_blocks(engines: Dict[str, Any], pdf_path: str, page_num: int) -> List[OCRBlock]:
    img_bytes = pdf_page_to_image(
        pdf_path, page_num,
        dpi=getattr(settings, "OCR_DPI", 150),
        max_dimension=getattr(settings, "OCR_MAX_DIMENSION", 1600),
    )
    if not img_bytes:
        logger.warning("rapid_ocr 跳過頁面 %s：轉圖失敗", page_num)
        return []
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    arr = np.asarray(img)

    lout = engines["layout"](arr)
    boxes = list(lout.boxes or [])
    names = list(lout.class_names or [])
    scores = list(lout.scores or [])

    blocks: List[OCRBlock] = []
    idx = 0
    for box, name, score in _reading_order(boxes, names, scores):
        name_l = (name or "").lower()
        if name_l in _SKIP_LABELS:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        pad = 6
        crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))

        if name_l in _TABLE_LABELS:
            try:
                html = _html_of(engines["table"](crop))
            except Exception as exc:
                logger.warning("rapid_ocr 第 %s 頁表格辨識失敗 (%s)", page_num, exc)
                html = ""
            md = _html_table_to_markdown(html) if html else ""
            if md:
                idx += 1
                blocks.append(OCRBlock(
                    block_type="table", text=md, page=page_num, paragraph_index=idx,
                    html=html, markdown=md, metadata={"ocr_engine": "rapid", "layout": name_l},
                ))
            continue

        text = _ocr_text(engines["text_ocr"], crop)
        if text:
            idx += 1
            blocks.append(OCRBlock(
                block_type="paragraph", text=text, page=page_num, paragraph_index=idx,
                metadata={"ocr_engine": "rapid", "layout": name_l},
            ))
    return blocks


def extract_image_pdf_blocks_rapid(
    pdf_path: str,
    *,
    tier: Optional[str] = None,
    version: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lightweight onnx OCR over an image PDF; same output schema as PP-StructureV3.

    Pass tier/version/device to override the configured defaults for ONE run
    (used by the per-doc high-accuracy re-OCR, which forces tier='server')
    without mutating global settings; such overrides use a one-off engine
    bundle rather than the cached singleton.
    """
    page_count = get_pdf_page_count(pdf_path)
    if page_count <= 0:
        return []
    if tier or version or device:
        d_tier, d_ver, d_dev = _resolve_cfg()
        engines = _build_engines(tier or d_tier, version or d_ver, device or d_dev)
    else:
        engines = _get_engines()
    all_blocks: List[OCRBlock] = []
    for page_num in range(1, page_count + 1):
        try:
            page_blocks = rapid_page_blocks(engines, pdf_path, page_num)
        except Exception as exc:
            logger.warning("rapid_ocr 第 %s 頁失敗 (%s)，跳過", page_num, exc)
            page_blocks = []
        all_blocks.extend(page_blocks)
        logger.info("rapid_ocr 完成第 %s 頁，抽出 %s 段", page_num, len(page_blocks))
    return normalize_ocr_blocks(all_blocks)
