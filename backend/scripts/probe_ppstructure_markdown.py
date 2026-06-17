"""Run new PPStructureV3-based extract on a single page; print resulting markdown."""
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ocr_pipeline import (
    _build_pp_structure,
    _replace_html_tables_with_markdown,
)
from app.services.pdf_image import pdf_page_to_image

import io
import numpy as np
from PIL import Image

PDF = Path(r"C:\Users\GU605_PR_MZ\Downloads\Rep_ORT(S)_FA506NC(NJXA)_2024 4_QCMC.xlsx (1).pdf")
PAGE = 3  # has a measurement-equipment table per earlier observation

print("Building PPStructureV3...")
pipe = _build_pp_structure()

print(f"Rendering page {PAGE}...")
img_bytes = pdf_page_to_image(str(PDF), PAGE, dpi=200, max_dimension=2200)
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
arr = np.array(img)

print("Predicting...")
result = pipe.predict(arr)

for page_result in result or []:
    md_dict = page_result._to_markdown(pretty=False)
    raw = (md_dict or {}).get("markdown_texts", "")
    print("\n===== RAW _to_markdown(pretty=False) =====")
    print(raw[:2000])
    print("\n===== AFTER _replace_html_tables_with_markdown =====")
    converted = _replace_html_tables_with_markdown(raw)
    print(converted[:3000])
    print(f"\n--- diagnostics ---")
    print(f"raw len: {len(raw)}, converted len: {len(converted)}")
    print(f"raw has <table>: {'<table' in raw.lower()}")
    print(f"converted has '|': {'|' in converted}")
    print(f"converted has '---': {'---' in converted}")
