"""引用溯源定位：回答「這段被引用的文字在第 N 頁的哪個位置」。

兩型 PDF 通吃，這是本功能存在的理由 —— 前端文字層高亮只對原生文字 PDF
有效，圖片型（掃描+OCR）的文字層是空的，永遠畫不出 <mark>：
  * 文字型：PyMuPDF page.search_for()，毫秒級
  * 圖片型：對「該頁」現場跑一次 RapidOCR 文字辨識（~4 秒），
    引擎逐行回傳座標框，比對 snippet 後回傳命中行的框

座標一律換算成 PDF point（72dpi 座標系），前端量測實際渲染尺寸後
自行縮放 —— 不猜 react-pdf 的渲染倍率。

不動入庫管線、不回填、不重建索引：成本只發生在使用者點「預覽」的當下。
"""
from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_MAX_RECTS = 60
_MIN_PHRASE = 8       # 片語太短會滿頁誤中（規範文件裡 "the test" 之類到處都是）
_MIN_OCR_LINE = 6     # OCR 行文字至少這麼長才拿去比對，同上理由


def _norm(s: str) -> str:
    """比對用正規化：去空白、簡轉繁、轉大寫。簡轉繁是必要的 —— OCR rec 模型
    常吐簡體（「介绍」），庫內 chunk 是 VL 產的繁體（「介紹」），不歸一整行落空。"""
    from .ollama_client import to_traditional
    return to_traditional(re.sub(r"\s+", "", s or "")).upper()


def locate(pdf_path: str, page_number: int, snippet: str) -> Dict[str, Any]:
    """回傳 {rects, page_width, page_height, source}。rects 為 PDF point 座標。"""
    import fitz

    with fitz.open(pdf_path) as doc:
        if not (1 <= page_number <= doc.page_count):
            return {"rects": [], "page_width": 0, "page_height": 0, "source": "none"}
        page = doc[page_number - 1]
        pw, ph = page.rect.width, page.rect.height

        # 逐字詞比對而非 search_for：實測本語料句點後常無空白（切不出句子片語），
        # search_for 又不跨行 ——「samples per second」跨行即 0 命中。改用 words
        # 座標 + 3-gram 滑窗：連續三個字詞（正規化串接）出現在 snippet 裡就標記，
        # 相鄰標記字詞併成一框。天然抗換行、空白與切塊邊界差異。
        words = page.get_text("words") or []
        target = _norm(snippet)
        if len(words) >= 5 and len(target) >= _MIN_PHRASE:
            toks = [(w[4], fitz.Rect(w[0], w[1], w[2], w[3]))
                    for w in words if str(w[4]).strip()]
            marked = [False] * len(toks)
            for i in range(len(toks) - 2):
                tri = _norm(toks[i][0] + toks[i + 1][0] + toks[i + 2][0])
                if len(tri) >= 6 and tri in target:
                    marked[i] = marked[i + 1] = marked[i + 2] = True
            rects: List[Dict[str, float]] = []
            run = None
            for (_txt, r), m in zip(toks, marked):
                if m:
                    merged = r if run is None else run | r
                    # 換行（合併後高度暴增）就先收掉這行，避免一框蓋整頁
                    if run is not None and merged.height > max(r.height, run.height) * 1.8:
                        rects.append({"x0": run.x0, "y0": run.y0, "x1": run.x1, "y1": run.y1})
                        run = r
                    else:
                        run = merged
                elif run is not None:
                    rects.append({"x0": run.x0, "y0": run.y0, "x1": run.x1, "y1": run.y1})
                    run = None
                if len(rects) >= _MAX_RECTS:
                    break
            if run is not None and len(rects) < _MAX_RECTS:
                rects.append({"x0": run.x0, "y0": run.y0, "x1": run.x1, "y1": run.y1})
            if rects:
                return {"rects": rects[:_MAX_RECTS], "page_width": pw,
                        "page_height": ph, "source": "text"}

    # 文字層空（圖片型）或片語全落空（OCR 入庫文字與 PDF 文字層歧異）→ OCR 定位
    return _locate_by_ocr(pdf_path, page_number, snippet, pw, ph)


def _locate_by_ocr(pdf_path: str, page_number: int, snippet: str,
                   pw: float, ph: float) -> Dict[str, Any]:
    from PIL import Image

    from .pdf_image import pdf_page_to_image
    from .rapid_ocr import _get_engines

    img_bytes = pdf_page_to_image(pdf_path, page_number, dpi=150, max_dimension=1600)
    if not img_bytes:
        return {"rects": [], "page_width": pw, "page_height": ph, "source": "none"}
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    scale = pw / img.size[0] if img.size[0] else 1.0   # 圖片 px → PDF pt

    try:
        res = _get_engines()["text_ocr"](np.asarray(img))
        # res.boxes 是 numpy 陣列 —— `arr or []` 會拋 truth-value ambiguous，
        # 不能用 or 慣用法
        _b, _t = getattr(res, "boxes", None), getattr(res, "txts", None)
        boxes = [] if _b is None else list(_b)
        txts = [] if _t is None else list(_t)
    except Exception as exc:  # noqa: BLE001
        logger.warning("locate OCR 失敗 %s p%s: %s", pdf_path, page_number, exc)
        return {"rects": [], "page_width": pw, "page_height": ph, "source": "none"}

    target = _norm(snippet)
    rects: List[Dict[str, float]] = []
    for box, txt in zip(boxes, txts):
        line = _norm(txt)
        # 行在 snippet 內，或（長行被 chunk 切斷時）行的前 12 字在 snippet 內
        if len(line) < _MIN_OCR_LINE:
            continue
        if line not in target and line[:12] not in target:
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        rects.append({"x0": min(xs) * scale, "y0": min(ys) * scale,
                      "x1": max(xs) * scale, "y1": max(ys) * scale})
        if len(rects) >= _MAX_RECTS:
            break
    return {"rects": rects, "page_width": pw, "page_height": ph,
            "source": "ocr" if rects else "none"}
