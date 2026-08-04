import os
import time
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document


class RAGPipelineOffline:
    # 图片格式（走 OCR）
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}

    def __init__(self, original_store, metadata_store, vector_store, clean_text_store, embed_model, archive_store=None):
        self.fs = original_store
        self.ms = metadata_store
        self.vs = vector_store
        self.ct = clean_text_store
        self.model = embed_model
        self.archive = archive_store

    # ========== OCR 引擎（延迟初始化） ==========
    _ocr_checked = False
    _ocr_available = False

    @classmethod
    def _check_ocr(cls) -> bool:
        """检测 OCR 依赖是否可用"""
        if cls._ocr_checked:
            return cls._ocr_available
        cls._ocr_checked = True
        try:
            import fitz  # pymupdf
            import pytesseract
            from PIL import Image
            cls._ocr_available = True
        except ImportError:
            cls._ocr_available = False
        return cls._ocr_available

    def _ocr_pdf(self, path: str) -> List[Document]:
        """PDF 扫描件 OCR：用 fitz 将每页渲染为图片，再 tesseract 识别"""
        import fitz
        import pytesseract
        from PIL import Image
        import io

        docs = []
        pdf = fitz.open(path)
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # 300 DPI 渲染
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page_number": page_num + 1,
                              "category": "OCR"}
                ))
        pdf.close()
        return docs

    def _ocr_image(self, path: str) -> List[Document]:
        """单张图片 OCR"""
        import pytesseract
        from PIL import Image

        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="chi_sim+eng").strip()
        if text:
            return [Document(
                page_content=text,
                metadata={"source": os.path.basename(path), "page_number": 1, "category": "OCR"}
            )]
        return []

    # ========== 文档加载 ==========
    def _loader(self, path):
        """统一文档加载器，按后缀选择最佳加载方式"""
        ext = os.path.splitext(path)[1].lower()

        # 图片 — 直接走 OCR
        if ext in self.IMAGE_EXTS:
            return None  # 特殊标记，在 _load() 中走 OCR

        # PDF — PyMuPDF（数字 PDF 直接读，扫描件后续 fallback）
        if ext == '.pdf':
            from langchain_community.document_loaders import PyMuPDFLoader
            return PyMuPDFLoader(path)

        # DOCX — Docx2txtLoader
        if ext in {'.docx', '.doc'}:
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(path)

        # TXT/MD — 纯文本
        if ext in {'.txt', '.md'}:
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path, encoding="utf-8")

        # 其余 — UnstructuredFileLoader 统一处理
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader
            return UnstructuredFileLoader(path, mode="elements")
        except ImportError:
            raise ValueError(f"不支持的文件类型: {ext}（请安装 unstructured 库）")

    def _load(self, path):
        ext = os.path.splitext(path)[1].lower()

        # 图片直接 OCR
        if ext in self.IMAGE_EXTS:
            if not self._check_ocr():
                raise RuntimeError("OCR 需要安装 pytesseract、pymupdf、Pillow。请运行: pip install pytesseract pymupdf Pillow")
            return self._ocr_image(path)

        loader = self._loader(path)
        docs = loader.load()
        for d in docs:
            d.metadata.setdefault("source", os.path.basename(path))

        # PDF OCR 回退：如果 PyMuPDF 提取的文字太少（<50 字符），可能是扫描件，走 OCR
        if ext == '.pdf':
            total_chars = sum(len(d.page_content.strip()) for d in docs)
            if total_chars < 50 and self._check_ocr():
                print(f"[OCR] PDF 文字量仅 {total_chars} 字符，切换 OCR 识别: {os.path.basename(path)}")
                ocr_docs = self._ocr_pdf(path)
                if ocr_docs:
                    return ocr_docs

        return docs

    # ========== 入库 ==========
    def ingest_file(self, file_stream: bytes, original_filename: str,
                    uploader_id: str = "system",
                    doc_code: str = "", category: str = "其他", tags: str = "") -> dict:
        # 1. 原始文件入库
        file_meta = self.fs.ingest(file_stream, original_filename, uploader_id,
                                   doc_code=doc_code, category=category, tags=tags)
        self.ms.insert_file(file_meta)
        path, file_id = file_meta["storage_path"], file_meta["file_id"]

        # 2. 加载 + 清洗链
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

        # 3. 构建 chunk 记录
        records, texts, ids, v_metas = [], [], [], []
        for idx, doc in enumerate(final):
            cid = f"{file_id}#chunk_{idx:04d}"
            m = doc.metadata
            records.append({"chunk_id": cid, "start_page": m.get("start_page"), "end_page": m.get("end_page"),
                           "char_len": len(doc.page_content), "level": m.get("level"), "category": m.get("category")})
            texts.append(doc.page_content)
            ids.append(cid)
            v_metas.append({"chunk_id": cid, "file_id": file_id, "level": m.get("level", "")})

        # 4. 落盘
        self.ct.append(file_id, final)
        self.ms.insert_chunks(file_id, records)
        embs = self.model.encode(texts, show_progress_bar=False)
        if hasattr(embs, 'cpu'):
            embs = embs.cpu().numpy()
        self.vs.add(ids, embs, v_metas)

        return {"file_id": file_meta["file_id"], "chunks_count": len(ids), "file_meta": file_meta}

    # ========== 带归档的更新 ==========
    def update_file(self, doc_code: str, new_stream: bytes, new_filename: str,
                    uploader_id: str = "system") -> dict:
        old_meta = self.ms.find_by_doc_code(doc_code)
        if not old_meta:
            raise ValueError(f"doc_code={doc_code} 没有对应的活跃文件，请先上传。")

        old_file_id = old_meta["file_id"]

        if self.archive:
            self.archive.archive(
                file_id=old_file_id,
                storage_path=old_meta["storage_path"],
                original_filename=old_meta["original_filename"]
            )

        self.ms.deactivate_file(old_file_id)
        self.vs.delete_by_file_id(old_file_id)

        return self.ingest_file(new_stream, new_filename, uploader_id, doc_code=doc_code)
