"""Run PPStructureV3 on one page of the EMC PDF, dump every block's full field map."""
import sys, os, io, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ocr_pipeline import _build_pp_structure
from app.services.pdf_image import pdf_page_to_image

import numpy as np
from PIL import Image

PDF = Path(r"C:\Users\GU605_PR_MZ\Downloads\Rep_ORT(S)_FA506NC(NJXA)_2024 4_QCMC.xlsx (1).pdf")
PAGE = 3  # page 3 has a table (per earlier observation)

print(f"Building PPStructureV3...")
pipe = _build_pp_structure()

print(f"Rendering page {PAGE}...")
img_bytes = pdf_page_to_image(str(PDF), PAGE, dpi=200, max_dimension=2200)
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
arr = np.array(img)

print(f"Predicting...")
result = pipe.predict(arr)

for page_idx, page_result in enumerate(result or []):
    print(f"\n=== page_result[{page_idx}] ===")
    print(f"type: {type(page_result).__name__}")

    # Try to list top-level keys
    if isinstance(page_result, dict):
        print(f"dict keys: {list(page_result.keys())[:30]}")
    else:
        attrs = [a for a in dir(page_result) if not a.startswith('_')][:40]
        print(f"object attrs: {attrs}")

    # Find the parsing_res_list / blocks
    blocks = None
    for key in ("parsing_res_list", "parsing_res", "layout_parsing_res", "blocks"):
        val = page_result.get(key) if isinstance(page_result, dict) else getattr(page_result, key, None)
        if isinstance(val, list) and val:
            blocks = val
            print(f"\nfound blocks under key='{key}', count={len(blocks)}")
            break

    if not blocks:
        # Try .json conversion
        try:
            j = page_result.json if hasattr(page_result, "json") else page_result
            if isinstance(j, dict):
                print(f"\nresult.json keys: {list(j.keys())[:30]}")
        except Exception:
            pass
        continue

    # Inspect each block (limit 5)
    for bi, blk in enumerate(blocks[:6]):
        print(f"\n--- block[{bi}] ---")
        print(f"  type: {type(blk).__name__}")
        if isinstance(blk, dict):
            for k, v in blk.items():
                vs = str(v)
                if len(vs) > 200:
                    vs = vs[:200] + "...[truncated]"
                print(f"  [{k}] = {vs}")
        else:
            for a in dir(blk):
                if a.startswith('_'): continue
                try:
                    v = getattr(blk, a)
                    if callable(v): continue
                    vs = str(v)
                    if len(vs) > 200:
                        vs = vs[:200] + "...[truncated]"
                    print(f"  .{a} = {vs}")
                except Exception:
                    pass
