"""OCR 子進程 worker（給 ocr_pipeline.extract_image_pdf_blocks_isolated 用）。

逐頁跑 PP-Structure，每頁完成就把結果 append 進 out_jsonl、並在 progress 寫 START/DONE。
某頁若 native segfault → 整個 worker process 死在那頁（progress 只有 START 沒 DONE），
父層據此把該頁加入 skip、重啟 worker 繼續剩下的頁。模型只在本 process 載入一次。

用法：python -m app.services.ocr_worker <pdf_path> <out_jsonl> <progress> <skip_csv>
"""
import io
import json
import os
import sys
from dataclasses import asdict


def main() -> None:
    pdf_path = sys.argv[1]
    out_jsonl = sys.argv[2]
    progress = sys.argv[3]
    skip_csv = sys.argv[4] if len(sys.argv) > 4 else ""
    skip = {int(x) for x in skip_csv.split(",") if x.strip()}

    from .pdf_image import get_pdf_page_count
    from . import ocr_pipeline as op

    # 測試鉤子：模擬指定頁 native segfault（START 後直接硬退出），用來驗證父層的崩潰恢復。
    crash_pages = {int(x) for x in os.environ.get("OCR_TEST_CRASH_PAGES", "").split(",") if x.strip()}

    n = get_pdf_page_count(pdf_path)
    pipeline = op._build_pp_structure()

    pf = io.open(progress, "a", encoding="utf-8")
    of = io.open(out_jsonl, "a", encoding="utf-8")
    try:
        for p in range(1, n + 1):
            if p in skip:
                continue
            pf.write(f"START {p}\n")
            pf.flush()
            if p in crash_pages:
                of.close()
                os._exit(139)  # 模擬 segfault（不寫 DONE → 父層判定崩潰頁）
            blocks = op._ppstructure_page_blocks(pipeline, pdf_path, p)  # 可能 segfault → process 死在這
            of.write(json.dumps(
                {"page": p, "blocks": [asdict(b) for b in blocks]}, ensure_ascii=False
            ) + "\n")
            of.flush()
            pf.write(f"DONE {p}\n")
            pf.flush()
        pf.write("ALL_DONE\n")
        pf.flush()
    finally:
        pf.close()
        of.close()


if __name__ == "__main__":
    main()
