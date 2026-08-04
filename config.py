import os

# ── Embedding 模式: "api" 或 "local" ──
# api:   通过 DeepSeek Embedding API（默认，无需本地模型）
# local: 使用本地 SentenceTransformer 模型
EMBED_MODE = os.getenv("EMBED_MODE", "api")

# 模型路径（仅本地模式需要）
EMBED_MODEL_PATH = r"E:\ana\MiniLM-L12-v2\sentence-transformers\paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_PATH = r"E:\ana\hf1\BAAI\bge-reranker-base"

# 存储路径
STORAGE_DIR = "storage"
ORIGINAL_STORE_PATH = os.path.join(STORAGE_DIR, "originals")
ARCHIVE_STORE_PATH = os.path.join(STORAGE_DIR, "archive")
VECTOR_STORE_PATH = os.path.join(STORAGE_DIR, "vectors")
METADATA_DB_PATH = os.path.join(STORAGE_DIR, "metadata.db")
CLEAN_TEXT_PATH = os.path.join(STORAGE_DIR, "clean_texts.db")

# 归档策略
ARCHIVE_TTL_DAYS = 30