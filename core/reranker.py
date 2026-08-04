"""
统一重排序层 — 支持 API 模式和本地模式
======================================
API 模式（默认）: 使用融合分数 (0.7*vector + 0.3*bm25) 代替 Cross-Encoder
本地模式: 使用 bge-reranker-base CrossEncoder
"""

import os
import json
from pathlib import Path
from typing import List, Dict


def _load_settings():
    settings_path = Path(__file__).resolve().parent.parent / ".rag_settings.json"
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


class ReRanker:
    """统一重排序接口"""

    def __init__(self, mode: str = None):
        """
        mode: "api" | "local" | None(自动检测)
        """
        settings = _load_settings()
        if mode is None:
            mode = settings.get("embed_mode", "api")

        self.mode = mode
        self._model = None

        if mode == "local":
            self._init_local(settings)

    def _init_local(self, settings: dict):
        """初始化本地 CrossEncoder"""
        model_path = settings.get("rerank_model_path", "")
        if not model_path:
            try:
                from config import RERANK_MODEL_PATH
                model_path = RERANK_MODEL_PATH
            except ImportError:
                pass

        if model_path and Path(model_path).exists():
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_path, max_length=512)
        else:
            # 本地模型不存在，降级到 API 模式
            self.mode = "api"

    def rerank(self, query: str, docs: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        docs: [{"text": ..., "score": ..., ...}, ...]
        返回 re-ranked 的 docs 列表（增加了 "rerank_score" 字段），取 top_n
        """
        if self.mode == "local" and self._model is not None:
            return self._rerank_local(query, docs, top_n)
        else:
            return self._rerank_fusion(docs, top_n)

    def _rerank_fusion(self, docs: List[Dict], top_n: int) -> List[Dict]:
        """
        融合分数重排序（无需模型）:
        向量分数占 70%，BM25 分数占 30%
        """
        for d in docs:
            vector_score = d.get("vector_score", d.get("score", 0))
            bm25_score = d.get("bm25_score", 0)
            d["rerank_score"] = vector_score * 0.7 + min(bm25_score / 10, 1.0) * 0.3

        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return docs[:top_n]

    def _rerank_local(self, query: str, docs: List[Dict], top_n: int) -> List[Dict]:
        """CrossEncoder 重排序"""
        pairs = [(query, d["text"]) for d in docs]
        scores = self._model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return docs[:top_n]
