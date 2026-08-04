import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pathlib import Path
from typing import List, Dict, Any, Optional


class HybridRetriever:
    """混合检索：向量 + FTS5 关键词 + 重排序。FTS5 替代了旧的内存 BM25。"""

    def __init__(self, vector_store, metadata_store, clean_text_store, embed_model,
                 rerank_model_path=None, rerank_model_name="BAAI/bge-reranker-v2-m3",
                 reranker=None):
        self.vs = vector_store
        self.ms = metadata_store
        self.ct = clean_text_store
        self.model = embed_model
        self.rerank_model_path = rerank_model_path
        self.rerank_model_name = rerank_model_name
        self._cross_encoder = None
        self._ext_reranker = reranker  # 外部 ReRanker 对象（API 模式）

    def _get_reranker(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            if self.rerank_model_path and Path(self.rerank_model_path).exists():
                self._cross_encoder = CrossEncoder(self.rerank_model_path, max_length=512)
            else:
                self._cross_encoder = CrossEncoder(self.rerank_model_name, max_length=512)
        return self._cross_encoder

    def retrieve(self, query: str, top_k=50, top_n=10, filter_file_ids=None, filter_levels=None, alpha=0.7):
        # ── 1. 向量检索 ──
        raw = self.model.encode(query)
        if hasattr(raw, 'cpu'):
            q_emb = raw.cpu().numpy()
        else:
            q_emb = raw
        if q_emb.ndim == 2:
            q_emb = q_emb[0]
        q_emb = q_emb.tolist()

        r = self.vs.query(
            q_emb, top_k=top_k * 2,
            file_id=filter_file_ids[0] if filter_file_ids and len(filter_file_ids) == 1 else None
        )

        v_ids = r["ids"][0] if r.get("ids") else []
        v_scores = {cid: max(0.0, 1.0 - d) for cid, d in zip(r["ids"][0], r["distances"][0])} if r.get(
            "distances") else {}

        # ── 2. FTS5 关键词检索 ──
        b_scores = self.ct.bm25_search(query, top_k=top_k * 2)

        # ── 3. 候选集合并 ──
        allowed = set(self.ms.filter_chunk_ids(file_ids=filter_file_ids, levels=filter_levels))
        candidates = (set(v_scores) | set(b_scores))
        if allowed:
            candidates &= allowed

        # ── 4. 分数融合 ──
        def norm(s):
            if not s:
                return {}
            vals = list(s.values())
            return {k: (v - min(vals)) / (max(vals) - min(vals)) if max(vals) > min(vals) else 0.5
                    for k, v in s.items()}

        nv, nb = norm(v_scores), norm(b_scores)
        fused = {cid: alpha * nv.get(cid, 0) + (1 - alpha) * nb.get(cid, 0) for cid in candidates}

        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        texts = self.ct.fetch(top_ids)

        docs_in = [{"chunk_id": cid, "text": texts[cid], "score": fused[cid]} for cid in top_ids if cid in texts]

        # ── 5. 重排序 ──
        reranked = self._rerank(query, docs_in, top_n=top_n)

        # ── 6. 附加元数据 ──
        for d in reranked:
            cm = self.ms.get_chunk_meta(d["chunk_id"])
            fm = self.ms.get_file(cm["file_id"]) if cm else None
            d["meta"] = {"file_id": cm["file_id"] if cm else None, "filename": fm["original_filename"] if fm else None}
        return reranked

    def _rerank(self, query, docs, top_n):
        if self._ext_reranker is not None:
            return self._ext_reranker.rerank(query, docs, top_n)
        reranker = self._get_reranker()
        pairs = [(query, d["text"]) for d in docs]
        scores = reranker.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs[:top_n]


# LangChain 适配层
class RAGHybridRetrieverLC(BaseRetriever):
    hybrid: object = None
    top_n: int = 5

    def _get_relevant_documents(self, query: str) -> List[Document]:
        hits = self.hybrid.retrieve(query=query, top_n=self.top_n)
        return [Document(page_content=hit["text"],
                         metadata={"chunk_id": hit["chunk_id"], "source": hit["meta"]["filename"],
                                   "score": hit.get("rerank_score", 0)}) for hit in hits]
