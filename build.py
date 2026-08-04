"""
系统装配 — 根据配置选择 API 或本地模式
=========================================
API 模式: DeepSeek Embedding API（默认，无需本地模型）
本地模式: 本地 SentenceTransformer + CrossEncoder 模型
"""

from core.stores import OriginalFileStore, MetadataStore, VectorStore, CleanTextStore, ArchiveStore
from core.pipeline import RAGPipelineOffline
from core.retriever import HybridRetriever, RAGHybridRetrieverLC
from tools.ingest import RAGIngestTool, RAGIngestFolderTool
from tools.retrieve import RAGRetrieveTool
from tools.manage import RAGUpdateTool, RAGListArchiveTool, RAGListActiveTool
from config import *


def build():
    # ── 1. 嵌入模型 ──
    from core.embedder import Embedder
    embedder = Embedder(mode=EMBED_MODE)
    print(f"[build] Embedding 模式: {embedder.mode}, 维度: {embedder.dim}")

    # ── 2. 存储层 ──
    fs = OriginalFileStore(ORIGINAL_STORE_PATH)
    ms = MetadataStore(METADATA_DB_PATH)
    vs = VectorStore(VECTOR_STORE_PATH)
    ct = CleanTextStore(CLEAN_TEXT_PATH)
    archive = ArchiveStore(ARCHIVE_STORE_PATH, ARCHIVE_TTL_DAYS)

    # ── 3. 入库管道 ──
    pipeline = RAGPipelineOffline(fs, ms, vs, ct, embedder, archive)

    # ── 4. 重排序器 ──
    from core.reranker import ReRanker
    reranker = ReRanker(mode=EMBED_MODE)
    print(f"[build] Reranker 模式: {reranker.mode}")

    # ── 5. 混合检索器 ──
    hybrid = HybridRetriever(
        vector_store=vs,
        metadata_store=ms,
        clean_text_store=ct,
        embed_model=embedder,
        rerank_model_path=RERANK_MODEL_PATH if EMBED_MODE == "local" else None,
        reranker=reranker,
    )
    retriever_lc = RAGHybridRetrieverLC(hybrid=hybrid)

    # ── 6. 工具列表 ──
    tools = [
        RAGIngestTool(pipeline=pipeline),
        RAGIngestFolderTool(pipeline=pipeline),
        RAGRetrieveTool(retriever=retriever_lc),
        RAGUpdateTool(pipeline=pipeline),
        RAGListArchiveTool(archive_store=archive),
        RAGListActiveTool(metadata_store=ms),
    ]
    return retriever_lc, tools
