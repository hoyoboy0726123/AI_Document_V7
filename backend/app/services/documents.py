import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session, selectinload

from .. import models
from ..core.config import settings
from . import ai, vector_store
from .ocr_pipeline import extract_image_pdf_blocks
from .pdf_processing import extract_text_and_segments, split_segments_into_chunks
from .system_config import SystemConfigService


def _generate_faiss_id() -> int:
    return int(uuid.uuid4().int % (1 << 63))


class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self._pdf_storage = Path(settings.PDF_STORAGE_DIR)
        self._pdf_temp = Path(settings.PDF_TEMP_DIR)
        self._pdf_storage.mkdir(parents=True, exist_ok=True)
        self._pdf_temp.mkdir(parents=True, exist_ok=True)

    # ---- Retrieval helpers ----
    def get(self, document_id: str) -> Optional[models.Document]:
        return (
            self.db.query(models.Document)
            .options(selectinload(models.Document.classification))
            .filter(models.Document.id == document_id)
            .first()
        )

    def list(
        self,
        *,
        page: int,
        page_size: int,
        search_term: Optional[str] = None,
        classification_id: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        folder_id: Optional[str] = None,
        folder_ids: Optional[List[str]] = None,
    ) -> Tuple[List[models.Document], int]:
        query = (
            self.db.query(models.Document)
            .options(
                selectinload(models.Document.classification),
            )
        )

        if search_term:
            like_pattern = f"%{search_term}%"
            query = query.filter(models.Document.title.ilike(like_pattern))

        if classification_id:
            query = query.filter(models.Document.classification_id == classification_id)

        # folder_ids takes precedence (recursive: selected folder + all descendants)
        if folder_ids is not None:
            query = query.filter(models.Document.folder_id.in_(folder_ids))
        elif folder_id is not None:
            if folder_id == "__root__":
                query = query.filter(models.Document.folder_id == None)  # noqa: E711
            else:
                query = query.filter(models.Document.folder_id == folder_id)

        documents = query.order_by(models.Document.created_at.desc()).all()

        if metadata_filters:
            documents = [
                doc
                for doc in documents
                if self._metadata_match(doc.metadata_data or {}, metadata_filters)
            ]

        total = len(documents)
        start = max(page - 1, 0) * page_size
        end = start + page_size
        return documents[start:end], total

    # ---- ????? ----
    def create(
        self,
        *,
        title: str,
        content: Optional[str],
        metadata: Dict[str, Any],
        creator: models.User,
        classification: Optional[models.ClassificationCategory] = None,
        pdf_temp_path: Optional[str] = None,
        segments: Optional[List[Dict[str, Any]]] = None,
        ai_summary: Optional[str] = None,
        is_image_based: bool = False,
        original_filename: Optional[str] = None,
        force_vision: bool = False,
        force_ocr: bool = False,
    ) -> models.Document:
        document = models.Document(
            title=title,
            content=content,
            creator_id=creator.id,
            created_at=datetime.utcnow(),
            ai_summary=ai_summary,
            is_image_based=is_image_based,
            ocr_status="pending" if (is_image_based or force_ocr) else "not_needed",
        )
        document.metadata_data = metadata or {}
        if classification is not None:
            document.classification_id = classification.id
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document, attribute_names=["classification"])

        if pdf_temp_path:
            self._finalize_pdf_and_index(
                document,
                pdf_temp_path,
                segments=segments,
                original_filename=original_filename,
                force_vision=force_vision,
                force_ocr=force_ocr,
            )

        return document

    def update(
        self,
        document: models.Document,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ai_summary: Optional[str] = None,
        classification: Optional[models.ClassificationCategory] = None,
        pdf_temp_path: Optional[str] = None,
    ) -> models.Document:
        if title is not None:
            document.title = title
        if content is not None:
            document.content = content
        if metadata is not None:
            document.metadata_data = metadata or {}
        if ai_summary is not None:
            document.ai_summary = ai_summary
        if classification is not None:
            document.classification_id = classification.id
        self.db.commit()
        self.db.refresh(document, attribute_names=["classification"])

        if pdf_temp_path:
            self._finalize_pdf_and_index(document, pdf_temp_path)

        return document

    def validate_metadata(self, metadata: Dict[str, Any], required_fields: List[models.MetadataField]) -> None:
        errors: List[str] = []
        metadata = metadata or {}

        for field in required_fields:
            value = metadata.get(field.name)

            if field.is_required and (value is None or (isinstance(value, str) and value.strip() == "")):
                errors.append(f"Missing required metadata: {field.display_name}")
                continue

            if value is None:
                continue

            try:
                if field.field_type in {"text", "textarea"}:
                    if not isinstance(value, str):
                        raise ValueError("must be a string")
                elif field.field_type == "number":
                    if not isinstance(value, (int, float)):
                        raise ValueError("must be a number")
                elif field.field_type == "date":
                    if isinstance(value, str):
                        datetime.fromisoformat(value)
                    else:
                        raise ValueError("must be an ISO date string")
                elif field.field_type == "select":
                    valid_values = {opt.value for opt in field.options if opt.is_active}
                    if value not in valid_values:
                        raise ValueError(f"value must be one of: {', '.join(valid_values)}")
                elif field.field_type == "multi_select":
                    if not isinstance(value, list):
                        raise ValueError("must be a list")
                    valid_values = {opt.value for opt in field.options if opt.is_active}
                    invalid = [item for item in value if item not in valid_values]
                    if invalid:
                        raise ValueError(f"invalid selection(s): {', '.join(invalid)}")
            except ValueError as exc:
                errors.append(f"{field.display_name} {exc}")

        if errors:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=errors)

    def apply_classification(
        self,
        document: models.Document,
        *,
        classification: models.ClassificationCategory,
    ) -> models.Document:
        document.classification_id = classification.id
        self.db.commit()
        self.db.refresh(document)
        return document

    def update_metadata_fields(self, document: models.Document, *, changes: Dict[str, Any]) -> models.Document:
        document.metadata_data = {**(document.metadata_data or {}), **(changes or {})}
        self.db.commit()
        self.db.refresh(document, attribute_names=["classification"])
        return document

    def delete(self, document: models.Document) -> None:
        """刪除文件及其相關資料（chunks、向量、PDF 檔案）"""
        # 1. 刪除向量存儲中的 embeddings
        existing_chunks = (
            self.db.query(models.DocumentChunk)
            .filter(models.DocumentChunk.document_id == document.id)
            .all()
        )
        existing_faiss_ids = [chunk.faiss_id for chunk in existing_chunks if chunk.faiss_id]
        if existing_faiss_ids:
            vector_store.remove_embeddings(existing_faiss_ids)

        # 2. 刪除資料庫中的 chunks
        self.db.query(models.DocumentChunk).filter(
            models.DocumentChunk.document_id == document.id
        ).delete()

        # 3. 刪除 PDF 檔案（使用安全的路徑驗證）
        if document.pdf_path:
            from ..utils.security import safe_file_delete
            from ..core.config import settings
            try:
                safe_file_delete(document.pdf_path, settings.FILE_STORAGE_DIR)
            except Exception as e:
                # Log error but don't fail deletion if file removal fails
                logger.warning(f"Failed to delete PDF file {document.pdf_path}: {e}")

        # 4. 刪除文件記錄
        self.db.delete(document)
        self.db.commit()

    # ---- ???? ----
    def _metadata_match(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, expected in filters.items():
            value = metadata.get(key)
            if value is None:
                return False
            if isinstance(expected, list):
                if isinstance(value, list):
                    if not set(expected).issubset(set(value)):
                        return False
                else:
                    return False
            else:
                if isinstance(value, list):
                    if expected not in value:
                        return False
                elif value != expected:
                    return False
        return True

    def _resolve_temp_pdf(self, pdf_temp_path: str) -> Path:
        candidate = Path(pdf_temp_path)
        if candidate.exists():
            return candidate
        alt = self._pdf_temp / candidate.name
        if alt.exists():
            return alt
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Temporary PDF file not found or already removed",
        )

    def _finalize_pdf_and_index(
        self,
        document: models.Document,
        pdf_temp_path: str,
        segments: Optional[List[Dict[str, Any]]] = None,
        original_filename: Optional[str] = None,
        force_vision: bool = False,
        force_ocr: bool = False,
    ) -> None:
        temp_path = self._resolve_temp_pdf(pdf_temp_path)

        # Construct final filename: [ID]_[OriginalName] or [ID].pdf
        if original_filename:
            # Sanitize filename
            safe_name = "".join(c for c in original_filename if c.isalnum() or c in "._- ")
            final_name = f"{document.id}_{safe_name}"
            if not final_name.lower().endswith(".pdf"):
                final_name += ".pdf"
        else:
            final_name = f"{document.id}.pdf"

        final_path = self._pdf_storage / final_name
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_path), final_path)
        document.pdf_path = str(final_path)
        self.db.commit()

        self._rebuild_document_chunks(
            document,
            final_path,
            segments=segments,
            force_vision=force_vision,
            force_ocr=force_ocr,
        )

    def _rebuild_document_chunks(
        self,
        document: models.Document,
        pdf_path: Path,
        segments: Optional[List[Dict[str, Any]]] = None,
        force_vision: bool = False,
        force_ocr: bool = False,
        ocr_engine: Optional[str] = None,
        ocr_tier: Optional[str] = None,
    ) -> None:
        existing_chunks = (
            self.db.query(models.DocumentChunk)
            .filter(models.DocumentChunk.document_id == document.id)
            .all()
        )
        existing_faiss_ids = [chunk.faiss_id for chunk in existing_chunks]
        if existing_faiss_ids:
            vector_store.remove_embeddings(existing_faiss_ids)
            self.db.query(models.DocumentChunk).filter(models.DocumentChunk.document_id == document.id).delete()
            self.db.commit()

        # 強制 OCR 時即使是文字型 PDF 也走 PaddleOCR；忽略前端傳來的 segments
        if force_ocr:
            segments = None

        # 如果已經有 segments，就不需要重複提取
        text = None
        if segments is None:
            if document.is_image_based or force_ocr:
                document.ocr_status = "processing"
                document.ocr_method = "paddleocr"
                self.db.commit()

                try:
                    ocr_blocks = extract_image_pdf_blocks(
                        str(pdf_path), engine=ocr_engine, tier=ocr_tier
                    )
                    segments = [
                        {
                            "page": block.get("page"),
                            "paragraph_index": block.get("paragraph_index"),
                            "text": block.get("markdown") or block.get("html") or block.get("text", ""),
                        }
                        for block in ocr_blocks
                        if (block.get("markdown") or block.get("html") or block.get("text"))
                    ]
                    text = "\n\n".join(seg.get("text", "") for seg in segments if seg.get("text"))
                    document.content = text[:20000] if text else None
                    document.ocr_status = "completed" if segments else "failed"
                    self.db.commit()
                except Exception:
                    document.ocr_status = "failed"
                    self.db.commit()
                    raise
            elif force_vision:
                # 強制用 VL 視覺模型逐頁解析
                from .pdf_image import get_pdf_page_count, pdf_pages_to_images
                document.ocr_method = "vision"
                self.db.commit()
                total_pages = get_pdf_page_count(str(pdf_path))
                page_numbers = list(range(1, total_pages + 1))
                # max_dimension=768 keeps image tokens well under Ollama's 4096 ctx —
                # avoids qwen2.5vl GGML_ASSERT bug in Ollama 0.13.x+ when tokens overflow ctx.
                image_bytes_list = pdf_pages_to_images(str(pdf_path), page_numbers, dpi=80, max_dimension=768)
                segments = ai.extract_text_with_vision(image_bytes_list, page_numbers)
                text = "\n\n".join(s.get("text", "") for s in segments if s.get("text"))
                if document.content is None or not document.content.strip():
                    document.content = text[:20000] if text else None
                self.db.commit()
            else:
                # 沒有提供 segments，需要從 PDF 提取
                pdf_bytes = pdf_path.read_bytes()
                text, segments = extract_text_and_segments(pdf_bytes)
                if document.content is None or not document.content.strip():
                    document.content = text[:20000] if text else None
        else:
            # segments 已提供，完全不需要重複提取 PDF
            # content 應該在 create 時就已經設置（從 upload 回傳的 text）
            if document.content is None or not document.content.strip():
                # 如果真的沒有 content，從 segments 組合出文字
                text = "\n".join(seg.get("text", "") for seg in segments if seg.get("text"))
                document.content = text[:20000] if text else None
            else:
                # content 已有值，使用它
                text = document.content

        # 從數據庫獲取向量配置
        config_service = SystemConfigService(self.db)
        vector_config = config_service.get_vector_config()

        chunk_payloads = split_segments_into_chunks(
            segments,
            max_chars=vector_config["max_chars"],
            overlap_chars=vector_config["overlap_chars"]
        )
        if not chunk_payloads and text:
            # Fallback：如果沒有 chunk_payloads，使用整段文字（截取到 max_chars）
            chunk_payloads = [
                {
                    "chunk_index": 0,
                    "page": None,
                    "paragraph_index": None,
                    "text": text[:vector_config["max_chars"]],
                }
            ]

        chunk_models: List[models.DocumentChunk] = []
        for payload in chunk_payloads:
            chunk = models.DocumentChunk(
                document_id=document.id,
                chunk_index=payload["chunk_index"],
                page=payload.get("page"),
                paragraph_index=payload.get("paragraph_index"),
                text=payload["text"],
                embedding=[],
                faiss_id=_generate_faiss_id(),
            )
            self.db.add(chunk)
            chunk_models.append(chunk)

        self.db.commit()
        self.db.refresh(document, attribute_names=["chunks"])

        if not chunk_models:
            return

        embeddings = ai.embed_texts([chunk.text for chunk in chunk_models])
        if len(embeddings) != len(chunk_models):
            raise RuntimeError("Embedding count does not match chunk count")

        faiss_mapping: Dict[int, List[float]] = {}
        for chunk, embedding in zip(chunk_models, embeddings):
            chunk.embedding = embedding
            faiss_mapping[chunk.faiss_id] = embedding
        self.db.commit()

        vector_store.add_embeddings(faiss_mapping)

    def _re_embed_existing_chunks(self, document: models.Document) -> None:
        """
        重新向量化現有的 chunks，不重新提取 PDF 文字

        這個方法只重新生成 embeddings，保留現有的文字內容
        適用於：切換 embedding 模型、清空向量值後重建
        """
        # 獲取現有的 chunks
        existing_chunks = (
            self.db.query(models.DocumentChunk)
            .filter(models.DocumentChunk.document_id == document.id)
            .order_by(models.DocumentChunk.chunk_index)
            .all()
        )

        if not existing_chunks:
            raise ValueError("此文件沒有現有的文字塊，無法重新向量化")

        # 從 FAISS 中刪除舊的 embeddings
        existing_faiss_ids = [chunk.faiss_id for chunk in existing_chunks if chunk.faiss_id]
        if existing_faiss_ids:
            vector_store.remove_embeddings(existing_faiss_ids)

        # 清空所有 chunks 的 embeddings
        for chunk in existing_chunks:
            chunk.embedding = []
        self.db.commit()

        # 重新生成 embeddings（使用現有的文字內容）
        embeddings = ai.embed_texts([chunk.text for chunk in existing_chunks])
        if len(embeddings) != len(existing_chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        # 更新 chunks 和 FAISS 索引
        faiss_mapping: Dict[int, List[float]] = {}
        for chunk, embedding in zip(existing_chunks, embeddings):
            chunk.embedding = embedding
            faiss_mapping[chunk.faiss_id] = embedding
        self.db.commit()

        vector_store.add_embeddings(faiss_mapping)


# ── 背景任務執行函式（獨立 DB Session）────────────────────────────────────────

def run_document_finalize_task(
    task_id: str,
    document_id: str,
    pdf_temp_path: str,
    force_vision: bool,
    segments_json: Optional[str],
    original_filename: Optional[str],
    force_ocr: bool = False,
) -> None:
    """
    供 FastAPI BackgroundTasks 呼叫：移動 PDF 並建立向量索引。
    使用獨立 DB Session，不依賴請求生命週期。
    force_vision=True 時改用 VL 視覺模型解析；force_ocr=True 時強制走 PaddleOCR。
    """
    import json as _json
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
        document = db.query(models.Document).filter_by(id=document_id).first()

        if not task or not document:
            return

        task.status = "running"
        if force_ocr:
            task.message = "正在使用 PaddleOCR 辨識..."
        elif force_vision:
            task.message = "正在使用 VL 視覺模型解析..."
        else:
            task.message = "正在建立向量索引..."
        db.commit()

        service = DocumentService(db)
        segments = _json.loads(segments_json) if segments_json else None

        service._finalize_pdf_and_index(
            document,
            pdf_temp_path,
            segments=segments,
            original_filename=original_filename,
            force_vision=force_vision,
            force_ocr=force_ocr,
        )

        task.status = "completed"
        task.progress = 100
        if force_ocr:
            task.message = "PaddleOCR 辨識與向量化完成"
        elif force_vision:
            task.message = "VL 解析與向量化完成"
        else:
            task.message = "向量化完成"
        db.commit()

        # 自動觸發 KG 抽取（settings.KG_AUTO_EXTRACT=True 時）。
        # 這裡 inline 跑而不是再開背景任務，因為 finalize 本身已在背景執行緒；
        # 失敗只記 log，不讓主任務變 failed。
        try:
            from ..core.config import settings as _settings
            if getattr(_settings, "KG_AUTO_EXTRACT", True):
                from . import kg_pipeline as _kg_pipeline
                kg_task = models.BackgroundTask(
                    task_type="kg_extract",
                    status="pending",
                    progress=0,
                    message="準備抽取規範引用...",
                    document_id=document_id,
                    creator_id=document.creator_id,
                )
                db.add(kg_task)
                db.commit()
                db.refresh(kg_task)
                _kg_pipeline.run_kg_extract_task(kg_task.id, document_id)
        except Exception as kg_exc:
            import logging
            logging.getLogger(__name__).warning(
                "Auto KG extraction failed for document %s: %s", document_id, kg_exc
            )

    except Exception as exc:
        try:
            task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
            if task:
                task.status = "failed"
                task.error = str(exc)[:500]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


# 保留舊名稱相容性
run_vl_vectorize_task = run_document_finalize_task


def run_document_reocr_task(
    task_id: str,
    document_id: str,
    ocr_engine: str = "rapid",
    ocr_tier: str = "server",
) -> None:
    """背景重跑單份文件的 OCR（高精度：預設 engine=rapid, tier=server），重建 chunks + 重新向量化。

    用獨立 DB Session；對既有 pdf_path 重跑，不需重新上傳。
    """
    from ..database import SessionLocal

    db = SessionLocal()
    try:
        task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
        document = db.query(models.Document).filter_by(id=document_id).first()
        if not task or not document:
            return

        pdf_path = getattr(document, "pdf_path", None)
        if not pdf_path or not Path(pdf_path).exists():
            task.status = "failed"
            task.error = "找不到原始 PDF，無法重跑 OCR"
            db.commit()
            return

        task.status = "running"
        task.message = f"正在以 {ocr_engine}/{ocr_tier} 高精度重跑 OCR..."
        db.commit()

        service = DocumentService(db)
        service._rebuild_document_chunks(
            document,
            Path(pdf_path),
            segments=None,
            force_ocr=True,
            ocr_engine=ocr_engine,
            ocr_tier=ocr_tier,
        )

        task.status = "completed"
        task.progress = 100
        task.message = f"高精度 OCR（{ocr_engine}/{ocr_tier}）重跑與向量化完成"
        db.commit()
    except Exception as exc:
        try:
            task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
            if task:
                task.status = "failed"
                task.error = str(exc)[:500]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def run_upload_analysis_task(task_id: str, file_path: str, filename: str) -> None:
    """
    背景執行 PDF 文字提取 + AI 建議分析。
    結果存入 BackgroundTask.result，前端 poll 後取用。
    """
    from ..database import SessionLocal
    from . import ai
    from .metadata import MetadataService
    from .pdf_processing import extract_text_and_segments, suggest_keywords

    db = SessionLocal()
    try:
        task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
        if not task:
            return

        task.status = "running"
        task.progress = 15
        task.message = "正在提取 PDF 文字..."
        db.commit()

        pdf_bytes = Path(file_path).read_bytes()
        text, segments = extract_text_and_segments(pdf_bytes)

        # 判斷是否為圖片型 PDF
        text_length = len(text.strip()) if text else 0
        is_image_based = text_length < 100

        if is_image_based:
            from .pdf_image import get_pdf_page_count
            try:
                total_pages = get_pdf_page_count(file_path)
            except Exception:
                total_pages = None
            task.result = {
                "text": text or "",
                "filename": filename,
                "segments": [],
                "suggested_metadata": {},
                "suggestion": {},
                "pdf_temp_path": file_path,
                "is_image_based": True,
                "total_pages": total_pages,
            }
            task.status = "completed"
            task.progress = 100
            task.message = "偵測到圖片型 PDF"
            db.commit()
            return

        task.progress = 45
        task.message = "正在生成 AI 摘要與建議..."
        db.commit()

        # 取得分類與專案供 AI 參考
        metadata_service = MetadataService(db)
        metadata_fields = metadata_service.list_fields(active_only=True)

        classifications = db.query(models.ClassificationCategory).filter_by(is_active=True).all()
        classification_prompt = [
            f"{c.name} ({c.code})" if c.code else c.name for c in classifications
        ]

        project_field = next((f for f in metadata_fields if f.name == "project_id"), None)
        project_options = getattr(project_field, "options", []) if project_field else []
        project_prompt = [o.display_value or o.value for o in project_options]

        text_for_ai = text[:20000] if text else ""
        segs_for_ai = [s.model_dump() for s in segments]

        try:
            suggestion_payload = ai.generate_document_suggestion(
                text=text_for_ai,
                classifications=classification_prompt,
                projects=project_prompt,
                segments=segs_for_ai,
            )
        except Exception:
            suggestion_payload = {}

        task.progress = 80
        task.message = "整理關鍵字與元數據..."
        db.commit()

        # 關鍵字合併
        kw_candidates = list(suggestion_payload.get("keywords", [])) + suggest_keywords(text or "")
        seen = set()
        merged_keywords = []
        for k in kw_candidates:
            if k and k.lower() not in seen:
                seen.add(k.lower())
                merged_keywords.append(k)

        suggested_metadata: Dict[str, Any] = {"keywords": merged_keywords}

        # 比對 file_type
        file_type_field = next((f for f in metadata_fields if f.name == "file_type"), None)
        if file_type_field:
            raw_ft = (suggestion_payload.get("metadata") or {}).get("file_type")
            for opt in getattr(file_type_field, "options", []):
                if raw_ft and raw_ft in (opt.value, opt.display_value):
                    suggested_metadata["file_type"] = opt.value
                    break

        # 比對 project
        if project_field:
            raw_proj = (suggestion_payload.get("metadata") or {}).get("project_id") or suggestion_payload.get("project")
            for opt in project_options:
                if raw_proj and raw_proj in (opt.value, opt.display_value):
                    suggested_metadata["project_id"] = opt.value
                    break

        task.result = {
            "text": text_for_ai,
            "filename": filename,
            "segments": segs_for_ai,
            "suggested_metadata": suggested_metadata,
            "suggestion": suggestion_payload or {},
            "pdf_temp_path": file_path,
            "is_image_based": False,
        }
        task.status = "completed"
        task.progress = 100
        task.message = "PDF 分析完成"
        db.commit()

    except Exception as exc:
        try:
            task = db.query(models.BackgroundTask).filter_by(id=task_id).first()
            if task:
                task.status = "failed"
                task.error = str(exc)[:500]
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
