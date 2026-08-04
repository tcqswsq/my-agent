import os
import time
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document

# 延迟导入，防止循环依赖
def _get_loaders():
    from langchain_community.document_loaders import (
        UnstructuredPDFLoader, Docx2txtLoader, TextLoader,
        UnstructuredExcelLoader, UnstructuredPowerPointLoader,
        UnstructuredHTMLLoader, JSONLoader
    )
    return locals()

class RAGPipelineOffline:
    def __init__(self, original_store, metadata_store, vector_store, clean_text_store, embed_model, archive_store=None):
        self.fs = original_store
        self.ms = metadata_store
        self.vs = vector_store
        self.ct = clean_text_store
        self.model = embed_model
        self.archive = archive_store

    def _loader(self, path):
        """统一文档加载器：优先使用专用加载器保证解析质量，其余用 UnstructuredFileLoader 兜底"""
        ext = os.path.splitext(path)[1].lower()

        # PDF — PyMuPDF：完全离线、速度快、中文友好
        if ext == '.pdf':
            from langchain_community.document_loaders import PyMuPDFLoader
            return PyMuPDFLoader(path)

        # DOCX — Docx2txtLoader：轻量，不需要 Word 依赖
        if ext in {'.docx', '.doc'}:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(path)

        # TXT/MD — 纯文本，指定 UTF-8 编码
        if ext in {'.txt', '.md'}:
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path, encoding="utf-8")

        # 其他所有格式 — UnstructuredFileLoader 统一处理
        # 支持: XLSX, CSV, PPTX, HTML, JSON 等
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader
            return UnstructuredFileLoader(path, mode="elements")
        except ImportError:
            raise ValueError(f"不支持的文件类型: {ext}（请安装 unstructured 库）")

    def _load(self, path):
        docs = self._loader(path).load()
        for d in docs: d.metadata.setdefault("source", os.path.basename(path))
        return docs

    def ingest_file(self, file_stream: bytes, original_filename: str,
                    uploader_id: str = "system",
                    doc_code: str = "") -> dict:
        # 1. 原始文件入库时透传 doc_code
        file_meta = self.fs.ingest(file_stream, original_filename, uploader_id, doc_code=doc_code)
        self.ms.insert_file(file_meta)
        path, file_id = file_meta["storage_path"], file_meta["file_id"]

        # Layer 2: 加载 + 清洗链
        from rag_system.utils.processors import enterprise, merge, fenkuai
        raw = self._load(path)
        cleaned = enterprise(raw)
        merged = merge(cleaned, max_len=800)
        final = fenkuai(merged, self.model, sim_threshold=0.75)
        if not final:
            raise ValueError(
                f"文件「{original_filename}」经清洗后无任何有效内容，已拒绝入库。\n"
                f"可能原因：文件为空、仅包含特殊字符或已被清洗规则过滤。\n"
                f"建议：请检查文件内容或适当放宽清洗规则。"
            )

        # 构建 chunk 记录
        records, texts, ids, v_metas = [], [], [], []
        for idx, doc in enumerate(final):
            cid = f"{file_id}#chunk_{idx:04d}"
            m = doc.metadata
            records.append({"chunk_id": cid, "start_page": m.get("start_page"), "end_page": m.get("end_page"), "char_len": len(doc.page_content), "level": m.get("level"), "category": m.get("category")})
            texts.append(doc.page_content)
            ids.append(cid)
            v_metas.append({"chunk_id": cid, "file_id": file_id, "level": m.get("level", "")})

        # 落盘
        self.ct.append(file_id, final)
        self.ms.insert_chunks(file_id, records)
        embs = self.model.encode(texts, show_progress_bar=False)
        # 兼容 tensor 和 numpy 返回
        if hasattr(embs, 'cpu'):
            embs = embs.cpu().numpy()
        self.vs.add(ids, embs, v_metas)

        return {"file_id": file_meta["file_id"], "chunks_count": len(ids), "file_meta": file_meta}

    # ========== 新增：带归档的更新 ==========
    def update_file(self, doc_code: str, new_stream: bytes, new_filename: str,
                    uploader_id: str = "system") -> dict:
        """
        用 doc_code 定位旧文件 → 归档 → 停用 → 入库新版
        大模型只需要告诉系统 doc_code + 新文件路径，不用记 file_id
        """
        # 1. 按业务编码找到旧版
        old_meta = self.ms.find_by_doc_code(doc_code)
        if not old_meta:
            raise ValueError(f"doc_code={doc_code} 没有对应的活跃文件，请先上传。")

        old_file_id = old_meta["file_id"]

        # 2. 归档旧文件全文
        if self.archive:
            self.archive.archive(
                file_id=old_file_id,
                storage_path=old_meta["storage_path"],
                original_filename=old_meta["original_filename"]
            )

        # 3. 停用旧版本 + 删旧向量
        self.ms.deactivate_file(old_file_id)
        self.vs.delete_by_file_id(old_file_id)

        # 4. 用同一个 doc_code 入库新版（version_tag 自动升级）
        return self.ingest_file(new_stream, new_filename, uploader_id, doc_code=doc_code)