import logging
from pathlib import Path
from sqlalchemy import inspect, text, Column, String, Integer, JSON
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from . import models
from .utils.logging_config import setup_logging, get_logger

# Setup structured logging
setup_logging(
    log_level="DEBUG",
    log_file=Path("logs/app.log"),
    enable_colors=True
)
logger = get_logger(__name__)

from .api.v1 import admin, agent, auth, documents, folders, kg, metadata, rag, tasks, vector_search
from .core.config import settings
from .database import SessionLocal, engine
from .services import users as user_service


models.Base.metadata.create_all(bind=engine)


def ensure_schema_updates() -> None:
    """
    Safely update database schema using SQLAlchemy Inspector.

    DEPRECATED: This function should be replaced with Alembic migrations.
    TODO: Initialize Alembic and create proper migration scripts.

    Security: Uses SQLAlchemy's Inspector API to avoid SQL injection risks
    from raw PRAGMA statements.
    """
    try:
        inspector = inspect(engine)

        # Check documents table columns
        doc_columns = {col['name'] for col in inspector.get_columns('documents')}

        with engine.begin() as connection:
            if "pdf_path" not in doc_columns:
                logger.info("Adding pdf_path column to documents table")
                # Using parameterized Column definitions (safer than raw SQL)
                connection.execute(text("ALTER TABLE documents ADD COLUMN pdf_path VARCHAR(512)"))

            # 添加 full_analysis 欄位（用於整份文件 AI 分析）
            if "full_analysis" not in doc_columns:
                logger.info("Adding full_analysis column to documents table")
                # Note: JSON column may need different syntax for different databases
                if engine.dialect.name == 'sqlite':
                    connection.execute(text("ALTER TABLE documents ADD COLUMN full_analysis TEXT"))
                else:  # PostgreSQL
                    connection.execute(text("ALTER TABLE documents ADD COLUMN full_analysis JSON"))

            # 添加 ai_summary 欄位（用於存儲 AI 自動生成的文件摘要）
            if "ai_summary" not in doc_columns:
                logger.info("Adding ai_summary column to documents table")
                connection.execute(text("ALTER TABLE documents ADD COLUMN ai_summary TEXT"))

            # 添加 OCR 相關欄位（用於處理圖片型PDF）
            if "is_image_based" not in doc_columns:
                logger.info("Adding is_image_based column to documents table")
                connection.execute(text("ALTER TABLE documents ADD COLUMN is_image_based BOOLEAN DEFAULT 0 NOT NULL"))

            if "ocr_status" not in doc_columns:
                logger.info("Adding ocr_status column to documents table")
                connection.execute(text("ALTER TABLE documents ADD COLUMN ocr_status VARCHAR(20) DEFAULT 'not_needed' NOT NULL"))

            if "ocr_method" not in doc_columns:
                logger.info("Adding ocr_method column to documents table")
                connection.execute(text("ALTER TABLE documents ADD COLUMN ocr_method VARCHAR(50)"))

            if "folder_id" not in doc_columns:
                logger.info("Adding folder_id column to documents table")
                connection.execute(text("ALTER TABLE documents ADD COLUMN folder_id VARCHAR(36) REFERENCES folders(id)"))

        # Check document_chunks table columns
        if inspector.has_table('document_chunks'):
            chunk_columns = {col['name'] for col in inspector.get_columns('document_chunks')}

            with engine.begin() as connection:
                if "faiss_id" not in chunk_columns:
                    logger.info("Adding faiss_id column to document_chunks table")
                    connection.execute(text("ALTER TABLE document_chunks ADD COLUMN faiss_id INTEGER"))

                    # Backfill faiss_id values using parameterized query
                    rows = connection.execute(text("SELECT id FROM document_chunks")).fetchall()
                    for index, (chunk_id,) in enumerate(rows, start=1):
                        connection.execute(
                            text("UPDATE document_chunks SET faiss_id = :faiss WHERE id = :chunk_id"),
                            {"faiss": index, "chunk_id": chunk_id},
                        )

        # Check background_tasks table columns
        if inspector.has_table('background_tasks'):
            bg_columns = {col['name'] for col in inspector.get_columns('background_tasks')}
            with engine.begin() as connection:
                if 'result' not in bg_columns:
                    logger.info("Adding result column to background_tasks table")
                    if engine.dialect.name == 'sqlite':
                        connection.execute(text("ALTER TABLE background_tasks ADD COLUMN result TEXT"))
                    else:
                        connection.execute(text("ALTER TABLE background_tasks ADD COLUMN result JSON"))

        # Check document_notes table columns
        if inspector.has_table('document_notes'):
            note_columns = {col['name'] for col in inspector.get_columns('document_notes')}
            with engine.begin() as connection:
                if "user_id" not in note_columns:
                    logger.info("Adding user_id column to document_notes table")
                    connection.execute(text("ALTER TABLE document_notes ADD COLUMN user_id VARCHAR(36) REFERENCES users(id)"))

        logger.info("Schema updates completed successfully")

    except Exception as e:
        logger.error(f"Schema update failed: {e}")
        # Don't raise - allow app to start even if schema update fails
        # This prevents blocking startup in case of minor issues


def ensure_default_admin() -> None:
    username = settings.DEFAULT_ADMIN_USERNAME
    password = settings.DEFAULT_ADMIN_PASSWORD
    email = settings.DEFAULT_ADMIN_EMAIL

    if not username or not password or not email:
        return

    db = SessionLocal()
    try:
        existing = user_service.get_user_by_username(db, username)
        if existing:
            updated = False
            if existing.role != "admin":
                existing.role = "admin"
                updated = True

            if not user_service.verify_password(password, existing.hashed_password):
                existing.hashed_password = user_service.hash_password(password)
                updated = True

            if updated:
                db.commit()
                print(f"[init] Default admin '{username}' updated.")
            return

        hashed_password = user_service.hash_password(password)
        admin_user = models.User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            role="admin",
            is_active=True,
        )
        db.add(admin_user)
        db.commit()
        print(f"[init] Default admin '{username}' created.")
    except Exception as exc:  # pragma: no cover
        db.rollback()
        print(f"[init] Failed to create default admin: {exc}")
    finally:
        db.close()


def apply_llm_overrides_from_db() -> None:
    """
    Copy LLM/embedding provider settings from system_configs into the in-memory
    settings object, so the provider factory sees DB values without DB access.
    Called on startup; admin endpoint also mutates `settings` directly on save.
    """
    try:
        from .services.system_config import SystemConfigService

        db = SessionLocal()
        try:
            svc = SystemConfigService(db)
            overrides = svc.get_llm_provider_config()
            mapping = {
                "llm.provider": "LLM_PROVIDER",
                "llm.model": "LLM_MODEL",
                "embedding.provider": "EMBEDDING_PROVIDER",
                "embedding.model": "EMBEDDING_MODEL",
                "vision.provider": "VISION_PROVIDER",
                "gemini.api_key": "GEMINI_API_KEY",
            }
            for k, attr in mapping.items():
                if k in overrides:
                    try:
                        setattr(settings, attr, overrides[k])
                    except Exception:
                        pass
            # VL 模型同步改 OLLAMA_VISION_MODEL,讓現有 services/ai.py 直接吃到
            if "vision.model" in overrides and overrides["vision.model"]:
                try:
                    setattr(settings, "VISION_MODEL", overrides["vision.model"])
                    setattr(settings, "OLLAMA_VISION_MODEL", overrides["vision.model"])
                except Exception:
                    pass
            # LLM / Embedding 模型同步到 legacy OLLAMA_* 設定。
            # services/ai.py 等舊呼叫點直接用 ollama_client.get_client()(讀 OLLAMA_LLM_MODEL /
            # OLLAMA_EMBED_MODEL),不走 llm_provider 抽象。admin 改模型時若不同步,舊路徑
            # (如 RAG 答案生成)會繼續用 .env 的舊模型而失敗。僅在 provider=ollama 時鏡像。
            try:
                if overrides.get("llm.provider", "ollama") == "ollama" and overrides.get("llm.model"):
                    setattr(settings, "OLLAMA_LLM_MODEL", overrides["llm.model"])
                if overrides.get("embedding.provider", "ollama") == "ollama" and overrides.get("embedding.model"):
                    setattr(settings, "OLLAMA_EMBED_MODEL", overrides["embedding.model"])
            except Exception:
                pass
        finally:
            db.close()
    except Exception as exc:
        logger.warning("apply_llm_overrides_from_db failed: %s", exc)


def apply_ocr_overrides_from_db() -> None:
    """Copy OCR engine settings from system_configs into in-memory settings on startup."""
    try:
        from .services.system_config import SystemConfigService

        db = SessionLocal()
        try:
            overrides = SystemConfigService(db).get_ocr_config()
            mapping = {
                "ocr.engine": "OCR_ENGINE",
                "ocr.model_tier": "OCR_MODEL_TIER",
                "ocr.version": "OCR_VERSION",
                "ocr.device": "OCR_DEVICE",
            }
            for k, attr in mapping.items():
                if k in overrides:
                    try:
                        setattr(settings, attr, overrides[k])
                    except Exception:
                        pass
        finally:
            db.close()
    except Exception as exc:
        logger.warning("apply_ocr_overrides_from_db failed: %s", exc)


ensure_schema_updates()
ensure_default_admin()
apply_llm_overrides_from_db()
apply_ocr_overrides_from_db()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="AI Document V3 API",
)

# Add rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
app.include_router(metadata.router, prefix="/api/v1", tags=["Metadata"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(vector_search.router, prefix="/api/v1/vector-search", tags=["VectorSearch"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["Folders"])
app.include_router(kg.router, prefix="/api/v1/kg", tags=["KnowledgeGraph"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Document Management System!"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
