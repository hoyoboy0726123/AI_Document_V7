"""PaddleOCR smoke test via extract_image_pdf_blocks (force 1 page)."""
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parents[1]
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from app.services import ocr_pipeline


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: test_ocr_smoke.py <pdf_path>")
        return 2

    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"ERROR not found: {pdf_path}")
        return 2

    # Force page count to 1 so we don't OCR all 30 pages in a smoke test.
    ocr_pipeline.get_pdf_page_count = lambda *_a, **_k: 1

    t = time.time()
    blocks = ocr_pipeline.extract_image_pdf_blocks(pdf_path)
    print(f"OCR_TIME {round(time.time()-t, 2)}s")
    print(f"BLOCKS {len(blocks)}")

    for i, b in enumerate(blocks[:5]):
        text = b.get("text", "")
        # safe ASCII repr to avoid Windows console encoding issues
        print(f"BLOCK_{i+1} page={b.get('page')} idx={b.get('paragraph_index')} len={len(text)}")

    if not blocks:
        print("WARN no blocks extracted")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
