"""End-to-end re-upload to trigger PPStructureV3 path.

Steps:
1. Login as admin
2. Delete existing doc fbf59366-... (if present)
3. POST /upload/ with the PDF (sync — fast)
4. POST /documents with is_image_based=True (sync — triggers _rebuild_document_chunks → extract_image_pdf_blocks → PPStructureV3)
"""
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8001/api/v1"
PDF_PATH = Path(
    r"C:\Users\GU605_PR_MZ\Downloads\Rep_ORT(S)_FA506NC(NJXA)_2024 4_QCMC.xlsx (1).pdf"
)
EXISTING_DOC_ID = "a226132f-2304-4b7c-b4fe-02eba9cc62e6"  # empty doc from last failed attempt


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def login() -> str:
    r = requests.post(
        f"{API}/auth/login",
        data={"username": "admin", "password": "Admin@123"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def delete_existing(token: str) -> None:
    if not EXISTING_DOC_ID:
        log("DELETE skipped (no existing doc id)")
        return
    r = requests.delete(f"{API}/documents/{EXISTING_DOC_ID}", headers=auth(token), timeout=30)
    log(f"DELETE existing doc: HTTP {r.status_code}")


def upload_preview(token: str) -> dict:
    with PDF_PATH.open("rb") as f:
        files = {"file": (PDF_PATH.name, f, "application/pdf")}
        r = requests.post(f"{API}/documents/upload/", headers=auth(token), files=files, timeout=120)
    r.raise_for_status()
    data = r.json()
    log(
        f"UPLOAD preview ok: is_image_based={data.get('is_image_based')} "
        f"total_pages={data.get('total_pages')} temp={data.get('pdf_temp_path')}"
    )
    return data


def create_document(token: str, preview: dict) -> dict:
    payload = {
        "title": "EMC Test Report (PPStructure retest)",
        "content": preview.get("text") or "",
        "metadata": {},
        "source_pdf_path": preview.get("pdf_temp_path"),
        "is_image_based": False,  # server preview 說不是 image-based，照實寫
        "force_ocr": True,        # 但強制走 OCR pipeline → 背景任務跑 PaddleOCR/PPStructure
        "original_filename": PDF_PATH.name,
        # 不傳 segments：讓 server 端 _rebuild_document_chunks 內部 force_ocr=True 時自動清掉
    }
    log("CREATE document — force_ocr=True，bg task 跑 PaddleOCR/PPStructure，立即回 task_id")
    r = requests.post(
        f"{API}/documents",
        headers={**auth(token), "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    log(f"CREATE response: HTTP {r.status_code}")
    if r.status_code >= 400:
        log(f"  body: {r.text[:500]}")
        r.raise_for_status()
    data = r.json()
    log(f"CREATE ok: id={data.get('id')} ocr_status={data.get('ocr_status')}")
    return data


if __name__ == "__main__":
    log("=== PPStructureV3 retest started ===")
    token = login()
    log("login ok")
    delete_existing(token)
    preview = upload_preview(token)
    if not preview.get("is_image_based"):
        log("WARNING: server says PDF is NOT image-based — OCR may not trigger")
    create_document(token, preview)
    log("=== done ===")
