"""混合檢索（Hybrid Retrieval）：向量 + BM25 關鍵字全文，用 RRF 融合。

動機：純向量檢索對「分散、靠關鍵字才找得到」的子項目 recall 很差
（例：查 "pressure test spec" 時，12.5 Prescale Film Pressure Test 的向量相似度
反而排在修訂紀錄頁之後）。BM25 關鍵字檢索能把含關鍵字的段落排到前面，
兩路結果用 Reciprocal Rank Fusion (RRF) 融合，取雙方互補的優點。

設計重點：
- FTS5 + trigram tokenizer：trigram 對 CJK 友善（≥3 字可命中），不需外部斷詞器。
- FTS 表存在「同一個 SQLite 檔」，rowid = faiss_id，與向量結果用同一把 key 融合。
- 用獨立 sqlite3 連線操作 FTS，不干擾 ORM 的 transaction。
- 索引「列數不符就重建」（涵蓋 ingest / 刪除 / 重抽），免去在 ingestion 各處埋同步點。
- 非 SQLite（如 Postgres）自動退回純向量，保留兩種部署模式。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from ..core.config import settings
from . import vector_store

logger = logging.getLogger(__name__)

_FTS_TABLE = "chunk_fts"
_BUILD_LOCK = threading.Lock()

# trigram tokenizer 需要 token 長度 ≥ 3 才能命中
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
_CJK_RUN_RE = re.compile(r"[一-鿿]{3,}")
# 小數/帶點數字整體保留（如 101.78、34.8500），讓「查表格某數值」能精準命中該列
# （trigram 對 "101.78" 會命中 "101.7800"）；否則 [A-Za-z0-9]{3,} 會把它拆成 "101" 等無鑑別力的詞。
_DECIMAL_RE = re.compile(r"\d+(?:\.\d+)+")


def _sqlite_path(db: Session) -> Optional[str]:
    """從 SQLAlchemy session 取得 SQLite 檔路徑；非 SQLite 回 None。"""
    try:
        bind = db.get_bind()
        if bind.dialect.name != "sqlite":
            return None
        path = bind.url.database
        if not path or path == ":memory:":
            return None
        return path
    except Exception:
        return None


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _ensure_index(conn: sqlite3.Connection, *, force: bool = False) -> bool:
    """確保 FTS 表存在且與 document_chunks 同步；回傳是否可用。"""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (_FTS_TABLE,)
    )
    exists = cur.fetchone() is not None
    if not exists:
        cur.execute(
            f"CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5(text, tokenize = 'trigram')"
        )

    src_n = cur.execute("SELECT count(*) FROM document_chunks").fetchone()[0]
    fts_n = (
        cur.execute(f"SELECT count(*) FROM {_FTS_TABLE}").fetchone()[0] if exists else 0
    )
    if force or not exists or src_n != fts_n:
        cur.execute(f"DELETE FROM {_FTS_TABLE}")
        cur.execute(
            f"INSERT INTO {_FTS_TABLE}(rowid, text) "
            f"SELECT faiss_id, text FROM document_chunks WHERE faiss_id IS NOT NULL"
        )
        conn.commit()
        logger.info("hybrid_search: FTS index rebuilt (%d rows)", src_n)
    return True


def _build_match(query: str) -> Optional[str]:
    """把使用者問題轉成 FTS5 MATCH 字串（各 term 以 OR 連接，提高 recall）。"""
    if not query:
        return None
    terms: List[str] = []
    # 帶小數的數字整體保留（高鑑別力，能獨指某一列）
    terms.extend(_DECIMAL_RE.findall(query))
    for tok in _EN_TOKEN_RE.findall(query):
        terms.append(tok.lower())
    terms.extend(_CJK_RUN_RE.findall(query))
    seen: List[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    if not seen:
        return None
    return " OR ".join('"%s"' % t.replace('"', "") for t in seen)


def keyword_search(db: Session, query: str, top_k: int) -> List[Tuple[int, float]]:
    """BM25 關鍵字檢索；回傳 [(faiss_id, bm25_score)]，bm25 越小越相關。"""
    if not getattr(settings, "RAG_HYBRID_SEARCH", True):
        return []
    path = _sqlite_path(db)
    if not path:
        return []
    match = _build_match(query)
    if not match:
        return []
    try:
        with _BUILD_LOCK:
            conn = _connect(path)
            try:
                _ensure_index(conn)
                rows = conn.execute(
                    f"SELECT rowid, bm25({_FTS_TABLE}) AS s "
                    f"FROM {_FTS_TABLE} WHERE {_FTS_TABLE} MATCH ? "
                    f"ORDER BY s LIMIT ?",
                    (match, top_k),
                ).fetchall()
            finally:
                conn.close()
        return [(int(r[0]), float(r[1])) for r in rows]
    except Exception as e:  # FTS 任何問題都退回純向量，絕不讓檢索整個壞掉
        logger.warning("hybrid_search keyword_search failed, falling back: %s", e)
        return []


def fuse(
    db: Session,
    query: str,
    embedding: List[float],
    candidate_k: int,
) -> Tuple[List[Tuple[int, Optional[float]]], Set[int]]:
    """向量 + 關鍵字 RRF 融合。

    回傳：
      - ordered: [(faiss_id, vector_score 或 None)]，依 RRF 分數由高到低
      - kw_hits: 出現在關鍵字結果中的 faiss_id 集合（供呼叫端放行 keyword-only 命中）
    """
    rrf_k = getattr(settings, "RAG_RRF_K", 60)
    vec = vector_store.search(embedding, candidate_k)
    kw = keyword_search(db, query, candidate_k)

    vec_rank: Dict[int, int] = {fid: i for i, (fid, _) in enumerate(vec)}
    kw_rank: Dict[int, int] = {fid: i for i, (fid, _) in enumerate(kw)}
    vec_score: Dict[int, float] = {fid: s for fid, s in vec}

    if not kw_rank:
        # 沒有關鍵字結果（非 SQLite / 無有效 term / FTS 失敗）→ 等同原本純向量
        return [(fid, vec_score.get(fid)) for fid, _ in vec], set()

    fids = set(vec_rank) | set(kw_rank)

    def _rrf(fid: int) -> float:
        s = 0.0
        if fid in vec_rank:
            s += 1.0 / (rrf_k + vec_rank[fid] + 1)
        if fid in kw_rank:
            s += 1.0 / (rrf_k + kw_rank[fid] + 1)
        return s

    ordered = sorted(fids, key=_rrf, reverse=True)
    return [(fid, vec_score.get(fid)) for fid in ordered], set(kw_rank)
