import os
import hashlib
import uuid
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager
from typing import List, Optional
import numpy as np
import sqlite3


# ======================= 原始文件存储 =======================
class OriginalFileStore:
    def __init__(self, root: str = "storage/originals"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def ingest(self, file_stream: bytes, original_filename: str,
               uploader_id: str = "system", source_tag: str = "upload",
               doc_code: str = "", category: str = "其他", tags: str = "") -> dict:
        file_id = f"f_{uuid.uuid4().hex[:12]}"
        content_hash = hashlib.sha256(file_stream).hexdigest()
        try:
            import magic
            mime_type = magic.from_buffer(file_stream, mime=True)
        except Exception:
            mime_type = "application/octet-stream"
        ext = Path(original_filename).suffix.lower()
        storage_path = self.root / f"{file_id}{ext}"
        with open(storage_path, "wb") as f:
            f.write(file_stream)
        return {
            "file_id": file_id,
            "doc_code": doc_code,
            "content_hash": f"sha256:{content_hash}",
            "storage_path": str(storage_path),
            "original_filename": original_filename,
            "file_size": len(file_stream),
            "mime_type": mime_type,
            "upload_time": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
            "status": "uploaded",
            "uploader_id": uploader_id,
            "source_tag": source_tag,
            "category": category,
            "tags": tags,
        }


# ======================= 元数据存储（SQLite） =======================
class MetadataStore:
    def __init__(self, db_path: str = "storage/metadata.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # 预定义分类
    BUILTIN_CATEGORIES = ["制度规章", "技术文档", "财务资料", "人力资源", "项目管理", "产品文档", "培训材料", "其他"]

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    doc_code TEXT DEFAULT '',
                    original_filename TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    mime_type TEXT,
                    file_size INTEGER,
                    upload_time TEXT,
                    uploader_id TEXT,
                    source_tag TEXT,
                    status TEXT DEFAULT 'uploaded',
                    is_active INTEGER DEFAULT 1,
                    version_tag TEXT DEFAULT 'v1.0',
                    category TEXT DEFAULT '其他',
                    tags TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    start_page INTEGER,
                    end_page INTEGER,
                    char_len INTEGER,
                    level TEXT,
                    category TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(file_id) REFERENCES files(file_id)
                );
            """)
            # 兼容旧库迁移：在创建索引之前先补列
            for col, default in [("category", "其他"), ("tags", "")]:
                try:
                    conn.execute(f"ALTER TABLE files ADD COLUMN {col} TEXT DEFAULT '{default}'")
                except sqlite3.OperationalError:
                    pass
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_files_active ON files(is_active);
                CREATE INDEX IF NOT EXISTS idx_files_doc_code ON files(doc_code);
                CREATE INDEX IF NOT EXISTS idx_files_category ON files(category);
                CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
            """)
            conn.commit()

    def insert_file(self, meta: dict):
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO files
                (file_id, doc_code, original_filename, storage_path, content_hash,
                 mime_type, file_size, upload_time, uploader_id, source_tag, status,
                 version_tag, category, tags)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                meta["file_id"], meta.get("doc_code", ""),
                meta["original_filename"], meta["storage_path"], meta["content_hash"],
                meta["mime_type"], meta["file_size"], meta["upload_time"],
                meta["uploader_id"], meta["source_tag"], meta["status"],
                meta.get("version_tag", "v1.0"),
                meta.get("category", "其他"),
                meta.get("tags", ""),
            ))
            conn.commit()

    def insert_chunks(self, file_id: str, chunks: List[dict]):
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks 
                   (chunk_id, file_id, start_page, end_page, char_len, level, category) 
                   VALUES (?,?,?,?,?,?,?)""",
                [(ch["chunk_id"], file_id, ch.get("start_page"), ch.get("end_page"),
                  ch.get("char_len"), ch.get("level"), ch.get("category")) for ch in chunks]
            )
            conn.commit()

    # ✅ 这次补的核心方法
    def get_file(self, file_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE file_id=?", (file_id,)).fetchone()
            return dict(row) if row else None

    def get_chunk_meta(self, chunk_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
            return dict(row) if row else None

    def filter_chunk_ids(self, file_ids: Optional[List[str]] = None,
                         levels: Optional[List[str]] = None) -> List[str]:
        clauses, params = [], []
        if file_ids:
            clauses.append(f"file_id IN ({','.join('?' * len(file_ids))})")
            params.extend(file_ids)
        if levels:
            clauses.append(f"level IN ({','.join('?' * len(levels))})")
            params.extend(levels)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            return [r["chunk_id"] for r in conn.execute(f"SELECT chunk_id FROM chunks {where}", params).fetchall()]

    def find_by_doc_code(self, doc_code: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE doc_code=? AND is_active=1",
                (doc_code,)
            ).fetchone()
            return dict(row) if row else None

    def get_active_files(self, category: str = "", tag: str = "") -> List[dict]:
        with self._connect() as conn:
            query = "SELECT file_id, doc_code, original_filename, upload_time, category, tags FROM files WHERE is_active=1"
            params = []
            if category:
                query += " AND category=?"
                params.append(category)
            if tag:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
            query += " ORDER BY upload_time DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def update_file_metadata(self, file_id: str, category: str = "", tags: str = "") -> bool:
        """更新文件的分类和标签"""
        updates, params = [], []
        if category:
            updates.append("category=?")
            params.append(category)
        if tags:
            updates.append("tags=?")
            params.append(tags)
        if not updates:
            return False
        params.append(file_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE files SET {', '.join(updates)} WHERE file_id=?", params)
            conn.commit()
        return True

    def get_category_stats(self) -> List[dict]:
        """各分类文件数统计"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM files WHERE is_active=1 GROUP BY category ORDER BY cnt DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def deactivate_file(self, file_id: str):
        with self._connect() as conn:
            conn.execute("UPDATE files SET is_active=0 WHERE file_id=?", (file_id,))
            conn.commit()
class VectorStore:
    def __init__(self, persist_dir: str = "storage/vectors",
                 collection: str = "rag_chunks"):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunk_ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        if not chunk_ids:
            raise ValueError("向量列表为空，文件内容可能被清洗规则过滤。")
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def delete_by_file_id(self, file_id: str):
        try:
            self.collection.delete(where={"file_id": file_id})
        except Exception:
            pass

    def query(self, query_embedding: List[float], top_k: int = 50, file_id: Optional[str] = None) -> dict:
        where = {"file_id": file_id} if file_id else None
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

# ======================= 清洗文本存储（SQLite + FTS5） =======================
class CleanTextStore:
    """文本存储 + 全文检索，基于 SQLite FTS5（替代 JSONL + 内存 BM25）"""

    def __init__(self, db_path: str = "storage/clean_texts.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS clean_texts (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_clean_texts_file ON clean_texts(file_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS texts_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize='unicode61',
                    prefix='2'
                );
            """)
            conn.commit()

    def append(self, file_id: str, chunks: List):
        with self._connect() as conn:
            for idx, doc in enumerate(chunks):
                cid = f"{file_id}#chunk_{idx:04d}"
                conn.execute(
                    "INSERT OR REPLACE INTO clean_texts (chunk_id, file_id, text) VALUES (?, ?, ?)",
                    (cid, file_id, doc.page_content)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO texts_fts (chunk_id, text) VALUES (?, ?)",
                    (cid, doc.page_content)
                )
            conn.commit()

    def fetch(self, chunk_ids: List[str]) -> Dict[str, str]:
        if not chunk_ids:
            return {}
        with self._connect() as conn:
            placeholders = ','.join(['?'] * len(chunk_ids))
            rows = conn.execute(
                f"SELECT chunk_id, text FROM clean_texts WHERE chunk_id IN ({placeholders})",
                chunk_ids
            ).fetchall()
        return {r["chunk_id"]: r["text"] for r in rows}

    def bm25_search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        """FTS5 关键词检索，返回 {chunk_id: bm25_score}（0~1，越大越相关）"""
        # 清洗查询字符串，每个词用 OR 连接
        safe = query.replace('"', '').replace("'", "")
        terms = ' OR '.join(t for t in safe.split() if len(t) >= 1)
        if not terms:
            return {}
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT chunk_id, rank FROM texts_fts WHERE texts_fts MATCH ? ORDER BY rank LIMIT ?",
                    (terms, top_k)
                ).fetchall()
            if not rows:
                return {}
            # FTS5 rank: 负值，越小越好。归一化到 [0, 1]，越大越好
            ranks = [r["rank"] for r in rows]
            min_r, max_r = ranks[-1], ranks[0]
            if max_r == min_r:
                return {r["chunk_id"]: 0.8 for r in rows}
            return {r["chunk_id"]: round(1.0 - (r["rank"] - min_r) / (max_r - min_r), 4) for r in rows}
        except Exception:
            return {}

    def delete_by_file_id(self, file_id: str):
        """删除文件对应的所有文本（FTS5 + 主表）"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM clean_texts WHERE file_id = ?", (file_id,)
            ).fetchall()
            for r in rows:
                conn.execute("DELETE FROM texts_fts WHERE chunk_id = ?", (r["chunk_id"],))
            conn.execute("DELETE FROM clean_texts WHERE file_id = ?", (file_id,))
            conn.commit()


# ======================= 归档存储 =======================
class ArchiveStore:
    def __init__(self, root: str = "storage/archive", ttl_days: int = 30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days

    def archive(self, file_id: str, storage_path: str, original_filename: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        target_dir = self.root / file_id
        target_dir.mkdir(parents=True, exist_ok=True)

        archive_path = target_dir / f"{ts}_{original_filename}"
        shutil.copy2(storage_path, archive_path)
        return str(archive_path)

    def list_versions(self, file_id: str) -> list:
        """列出某个文件的归档版本"""
        target_dir = self.root / file_id
        if not target_dir.exists():
            return []
        versions = sorted(target_dir.iterdir(), key=lambda p: p.name, reverse=True)
        return [f"  📦 {v.name}" for v in versions]

    def cleanup_expired(self):
        now = time.time()
        ttl_sec = self.ttl_days * 86400
        deleted = 0

        for root, _, files in os.walk(self.root):
            for f in files:
                fp = Path(root) / f
                if now - fp.stat().st_mtime > ttl_sec:
                    fp.unlink()
                    deleted += 1

        for root, dirs, _ in os.walk(self.root, topdown=False):
            for d in dirs:
                dp = Path(root) / d
                if dp.exists() and not any(dp.iterdir()):
                    dp.rmdir()

        return deleted