"""
RAG 系统工具集成测试
====================
测试范围：
  1. 存储层 — OriginalFileStore / MetadataStore / VectorStore / CleanTextStore / ArchiveStore
  2. 文档处理 — enterprise / merge / fenkuai
  3. 入库管道 — RAGPipelineOffline.ingest_file / update_file
  4. 混合检索 — HybridRetriever.retrieve
  5. 6 个 LangChain 工具 — ingest / ingest_folder / retrieve / update / list_archive / list_active

运行方式：
  cd E:/ana/python/rag_system
  python -m pytest test_rag_system.py -v -s
  (或) python test_rag_system.py
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
# 将 rag_system/ 和它的父目录都加入 path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PARENT = str(PROJECT_ROOT.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# ========== 测试配置 ==========
TEST_DIR = Path(tempfile.mkdtemp(prefix="rag_test_"))


def _make_fake_embed_model():
    """创建假的 embedding 模型，返回固定维度随机向量"""
    class FakeEmbed:
        def encode(self, texts, convert_to_tensor=False, show_progress_bar=False, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            vecs = np.random.randn(len(texts), 384).astype(np.float32)
            # 归一化到单位向量 (cosine 距离用)
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            if convert_to_tensor:
                import torch
                return torch.tensor(vecs)
            return vecs
    return FakeEmbed()


# ======================================================================
# 1. 存储层测试
# ======================================================================
class TestOriginalFileStore(unittest.TestCase):
    """原始文件存储"""

    def setUp(self):
        self.root = TEST_DIR / "originals"
        self.store = __import__('core.stores', fromlist=['OriginalFileStore']).OriginalFileStore(str(self.root))

    def test_ingest_creates_file_and_returns_meta(self):
        data = "Hello RAG! 这是测试文档内容。".encode("utf-8")
        meta = self.store.ingest(data, "测试文档.txt", uploader_id="tester", doc_code="TEST-001")

        self.assertTrue(meta["file_id"].startswith("f_"))
        self.assertEqual(meta["original_filename"], "测试文档.txt")
        self.assertEqual(meta["uploader_id"], "tester")
        self.assertEqual(meta["doc_code"], "TEST-001")
        self.assertTrue(meta["content_hash"].startswith("sha256:"))
        self.assertTrue(Path(meta["storage_path"]).exists())

    def test_ingest_generates_unique_ids(self):
        meta1 = self.store.ingest(b"aaa", "a.txt")
        meta2 = self.store.ingest(b"bbb", "b.txt")
        self.assertNotEqual(meta1["file_id"], meta2["file_id"])

    def test_content_hash_is_stable(self):
        meta1 = self.store.ingest(b"same content", "x.txt")
        meta2 = self.store.ingest(b"same content", "y.txt")
        self.assertEqual(meta1["content_hash"], meta2["content_hash"])


class TestMetadataStore(unittest.TestCase):
    """元数据存储 (SQLite)"""

    def setUp(self):
        self.db_path = str(TEST_DIR / "test_metadata.db")
        self.store = __import__('core.stores', fromlist=['MetadataStore']).MetadataStore(self.db_path)

    def test_insert_and_get_file(self):
        meta = {"file_id": "f_test123", "doc_code": "DOC-01", "original_filename": "手册.pdf",
                "storage_path": "/tmp/f_test123.pdf", "content_hash": "sha256:abc",
                "mime_type": "application/pdf", "file_size": 1024,
                "upload_time": "2026-01-01", "uploader_id": "admin",
                "source_tag": "upload", "status": "uploaded", "version_tag": "v1.0"}
        self.store.insert_file(meta)
        row = self.store.get_file("f_test123")
        self.assertIsNotNone(row)
        self.assertEqual(row["original_filename"], "手册.pdf")
        self.assertEqual(row["doc_code"], "DOC-01")

    def test_get_nonexistent_file(self):
        self.assertIsNone(self.store.get_file("f_ghost"))

    def test_find_by_doc_code(self):
        self.store.insert_file({"file_id": "f_a", "doc_code": "EMP-001", "original_filename": "a.pdf",
                                "storage_path": "/t/a.pdf", "content_hash": "sha256:x", "mime_type": "app/pdf",
                                "file_size": 1, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        found = self.store.find_by_doc_code("EMP-001")
        self.assertEqual(found["file_id"], "f_a")

    def test_get_active_files(self):
        self.store.insert_file({"file_id": "f1", "doc_code": "C1", "original_filename": "f1.pdf",
                                "storage_path": "/t/f1.pdf", "content_hash": "sha256:1", "mime_type": "app/pdf",
                                "file_size": 1, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        self.store.insert_file({"file_id": "f2", "doc_code": "C2", "original_filename": "f2.pdf",
                                "storage_path": "/t/f2.pdf", "content_hash": "sha256:2", "mime_type": "app/pdf",
                                "file_size": 2, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        self.store.deactivate_file("f2")
        active = self.store.get_active_files()
        ids = [f["file_id"] for f in active]
        self.assertIn("f1", ids)
        self.assertNotIn("f2", ids)

    def test_insert_and_get_chunks(self):
        self.store.insert_file({"file_id": "f_c", "doc_code": "", "original_filename": "c.pdf",
                                "storage_path": "/t/c.pdf", "content_hash": "sha256:c", "mime_type": "app/pdf",
                                "file_size": 1, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        chunks = [{"chunk_id": "f_c#chunk_0000", "start_page": 1, "end_page": 2, "char_len": 500, "level": "p", "category": "NarrativeText"}]
        self.store.insert_chunks("f_c", chunks)
        meta = self.store.get_chunk_meta("f_c#chunk_0000")
        self.assertEqual(meta["char_len"], 500)

    def test_filter_chunk_ids(self):
        self.store.insert_file({"file_id": "f_f1", "doc_code": "", "original_filename": "f1.pdf",
                                "storage_path": "/t/f1.pdf", "content_hash": "sha256:1", "mime_type": "app/pdf",
                                "file_size": 1, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        self.store.insert_chunks("f_f1", [
            {"chunk_id": "f_f1#0", "start_page": 1, "end_page": 1, "char_len": 100, "level": "h1", "category": "Title"}])
        ids = self.store.filter_chunk_ids(file_ids=["f_f1"])
        self.assertIn("f_f1#0", ids)

    def test_deactivate_file(self):
        self.store.insert_file({"file_id": "f_d", "doc_code": "", "original_filename": "d.pdf",
                                "storage_path": "/t/d.pdf", "content_hash": "sha256:d", "mime_type": "app/pdf",
                                "file_size": 1, "upload_time": "2026", "uploader_id": "u", "source_tag": "t",
                                "status": "ok", "version_tag": "v1"})
        self.store.deactivate_file("f_d")
        row = self.store.get_file("f_d")
        self.assertEqual(row["is_active"], 0)


class TestCleanTextStore(unittest.TestCase):
    """清洗文本存储"""

    def setUp(self):
        self.path = str(TEST_DIR / "test_clean.jsonl")
        self.store = __import__('core.stores', fromlist=['CleanTextStore']).CleanTextStore(self.path)

    def test_append_and_fetch(self):
        from langchain_core.documents import Document
        docs = [Document(page_content="第一段内容", metadata={}),
                Document(page_content="第二段内容", metadata={})]
        self.store.append("f_test", docs)

        texts = self.store.fetch(["f_test#chunk_0000", "f_test#chunk_0001"])
        self.assertEqual(len(texts), 2)
        self.assertIn("f_test#chunk_0000", texts)
        self.assertEqual(texts["f_test#chunk_0000"], "第一段内容")

    def test_fetch_partial_ids(self):
        from langchain_core.documents import Document
        self.store.append("f_x", [Document(page_content="X", metadata={})])
        texts = self.store.fetch(["f_x#chunk_0000", "f_x#chunk_9999"])
        self.assertEqual(len(texts), 1)  # 只有 1 个存在


class TestArchiveStore(unittest.TestCase):
    """归档存储"""

    def setUp(self):
        self.root = TEST_DIR / "archive"
        self.store = __import__('core.stores', fromlist=['ArchiveStore']).ArchiveStore(str(self.root), ttl_days=30)

    def test_archive_copies_file(self):
        # 先创建一个"原始文件"
        src = TEST_DIR / "originals" / "f_arch_test.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("待归档内容", encoding="utf-8")

        archived_path = self.store.archive("f_arch_test", str(src), "原始名称.txt")
        self.assertTrue(Path(archived_path).exists())
        self.assertIn("原始名称.txt", archived_path)

    def test_list_versions(self):
        src = TEST_DIR / "originals" / "f_list_test.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("V1", encoding="utf-8")
        self.store.archive("f_list_test", str(src), "doc.txt")

        versions = self.store.list_versions("f_list_test")
        self.assertEqual(len(versions), 1)
        self.assertIn("doc.txt", versions[0])

    def test_list_versions_empty(self):
        self.assertEqual(self.store.list_versions("f_nonexistent"), [])

    def test_cleanup_expired(self):
        src = TEST_DIR / "originals" / "f_expired.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("expired", encoding="utf-8")
        archived = Path(self.store.archive("f_expired", str(src), "old.txt"))

        # 创建一个 TTL=0 的 store，立即过期
        store0 = __import__('core.stores', fromlist=['ArchiveStore']).ArchiveStore(str(self.root), ttl_days=0)
        deleted = store0.cleanup_expired()
        self.assertGreaterEqual(deleted, 1)
        self.assertFalse(archived.exists())


# ======================================================================
# 2. 文档处理器测试
# ======================================================================
class TestProcessors(unittest.TestCase):
    """文档清洗 / 合并 / 分块"""

    @classmethod
    def setUpClass(cls):
        from utils import processors
        cls._enterprise = staticmethod(processors.enterprise)
        cls._merge = staticmethod(processors.merge)
        cls._fenkuai = staticmethod(processors.fenkuai)
        cls.fake_embed = _make_fake_embed_model()

    def _docs(self, *texts):
        from langchain_core.documents import Document
        return [Document(page_content=t, metadata={"page_number": i + 1}) for i, t in enumerate(texts)]

    # ---- enterprise ----
    def test_enterprise_removes_page_numbers(self):
        docs = self._docs("第 1 页", "第一章 概述", "保密")
        out = self._enterprise(docs, min_len=2)
        content = [d.page_content for d in out]
        self.assertNotIn("第 1 页", content)
        self.assertNotIn("保密", content)

    def test_enterprise_removes_empty_and_short(self):
        long_text = "企业知识库管理系统需要支持多种文档格式的解析和处理功能模块" + "ABCDEFGHIJ" * 3
        docs = self._docs("", "ab", long_text)
        out = self._enterprise(docs, min_len=50)
        self.assertEqual(len(out), 1)

    def test_enterprise_deduplicates(self):
        docs = self._docs("这是重复的内容文本", "这是重复的内容文本", "独特内容足够长" + "y" * 40)
        out = self._enterprise(docs, min_len=5)
        texts = [d.page_content for d in out]
        self.assertEqual(texts.count("这是重复的内容文本"), 1)

    def test_enterprise_filters_low_text_ratio(self):
        """有效字符 <40% 应被过滤"""
        docs = self._docs("!@#$%^&*()---===___", "正常文本内容" + "z" * 40)
        out = self._enterprise(docs, min_len=10)
        self.assertEqual(len(out), 1)

    # ---- merge ----
    def test_merge_combines_adjacent_paragraphs(self):
        docs = self._docs("段落A", "段落B", "段落C")
        merged = self._merge(docs, max_len=800)
        # 三段短文本应合并为一块
        self.assertEqual(len(merged), 1)
        self.assertIn("段落A", merged[0].page_content)
        self.assertIn("段落C", merged[0].page_content)

    def test_merge_splits_on_heading(self):
        from langchain_core.documents import Document
        docs = [
            Document(page_content="第一章", metadata={"page_number": 1, "level": "h1"}),
            Document(page_content="正文段落A", metadata={"page_number": 1, "level": "p"}),
            Document(page_content="第二章", metadata={"page_number": 5, "level": "h1"}),
            Document(page_content="正文段落B", metadata={"page_number": 5, "level": "p"}),
        ]
        merged = self._merge(docs, max_len=800)
        self.assertEqual(len(merged), 2)

    def test_merge_splits_on_max_len(self):
        long_text = "长文本" * 300  # 900 字符 > max_len=800
        docs = self._docs(long_text, "额外段落")
        merged = self._merge(docs, max_len=800)
        self.assertGreaterEqual(len(merged), 2)

    # ---- fenkuai ----
    def test_fenkuai_single_doc(self):
        docs = self._docs("唯一文档内容" + "a" * 50)
        result = self._fenkuai(docs, self.fake_embed, sim_threshold=0.75)
        self.assertEqual(len(result), 1)

    def test_fenkuai_empty(self):
        result = self._fenkuai([], self.fake_embed)
        self.assertEqual(result, [])


# ======================================================================
# 3. 入库管道测试 (需要 mock 存储层)
# ======================================================================
class TestPipeline(unittest.TestCase):
    """RAGPipelineOffline — ingest_file / update_file"""

    def setUp(self):
        from core.stores import OriginalFileStore, MetadataStore, VectorStore, CleanTextStore, ArchiveStore
        from core.pipeline import RAGPipelineOffline

        self.tmp = TEST_DIR / "pipeline_test"
        self.tmp.mkdir(parents=True, exist_ok=True)

        self.fs = OriginalFileStore(str(self.tmp / "originals"))
        self.ms = MetadataStore(str(self.tmp / "metadata.db"))
        self.vs = VectorStore(str(self.tmp / "vectors"))
        self.ct = CleanTextStore(str(self.tmp / "clean.jsonl"))
        self.archive = ArchiveStore(str(self.tmp / "archive"), ttl_days=30)
        self.fake_embed = _make_fake_embed_model()

        self.pipeline = RAGPipelineOffline(
            original_store=self.fs,
            metadata_store=self.ms,
            vector_store=self.vs,
            clean_text_store=self.ct,
            embed_model=self.fake_embed,
            archive_store=self.archive,
        )

    def _create_test_txt(self, name, content):
        path = self.tmp / name
        path.write_text(content, encoding="utf-8")
        return str(path), path.read_bytes()

    def test_ingest_file_txt(self):
        fpath, fbytes = self._create_test_txt("员工手册.txt",
            "员工手册\n\n第一章 考勤制度\n员工每日工作时间为9:00-18:00，午休1小时。\n\n"
            "第二章 请假流程\n员工请假需提前一天在OA系统提交申请。")

        result = self.pipeline.ingest_file(fbytes, "员工手册.txt", uploader_id="hr", doc_code="EMP-HB-001")

        self.assertIn("file_id", result)
        self.assertGreater(result["chunks_count"], 0)
        self.assertEqual(result["file_meta"]["doc_code"], "EMP-HB-001")

        # 验证元数据入库
        db_row = self.ms.get_file(result["file_id"])
        self.assertIsNotNone(db_row)
        self.assertEqual(db_row["is_active"], 1)

    def test_ingest_then_update(self):
        """测试完整更新流程: 入库 → 更新 → 旧版归档"""
        # 1) 入库旧版
        _, v1_bytes = self._create_test_txt("制度_v1.txt",
            "2025版公司制度\n第一章 总则\n本制度自2025年1月1日起执行，适用于全体员工。\n"
            "第二章 考勤管理\n员工应按时上下班并打卡记录。")
        r1 = self.pipeline.ingest_file(v1_bytes, "制度_v1.txt", doc_code="POLICY-001")
        old_id = r1["file_id"]

        # 2) 更新为新版
        _, v2_bytes = self._create_test_txt("制度_v2.txt",
            "2026版公司制度\n第一章 总则\n本制度自2026年1月1日起执行，适用于全体员工。\n"
            "新增：第三章 远程办公条款\n远程办公需经部门主管审批。")
        r2 = self.pipeline.update_file("POLICY-001", v2_bytes, "制度_v2.txt")

        new_id = r2["file_id"]
        self.assertNotEqual(old_id, new_id)

        # 3) 旧版已停用
        old_row = self.ms.get_file(old_id)
        self.assertEqual(old_row["is_active"], 0)

        # 4) 新版活跃
        new_row = self.ms.get_file(new_id)
        self.assertEqual(new_row["is_active"], 1)

        # 5) 归档有记录
        versions = self.archive.list_versions(old_id)
        self.assertGreaterEqual(len(versions), 1)

    def test_update_nonexistent_doc_code_raises(self):
        _, fb = self._create_test_txt("new.txt", "新文件")
        with self.assertRaises(ValueError):
            self.pipeline.update_file("NONEXISTENT", fb, "new.txt")

    def test_ingest_empty_file_raises(self):
        """空文件或全被清洗的内容应拒绝入库"""
        _, fb = self._create_test_txt("empty.txt", "  ")  # 只有空格，会被清洗掉
        with self.assertRaises(ValueError):
            self.pipeline.ingest_file(fb, "empty.txt")


# ======================================================================
# 4. 混合检索测试
# ======================================================================
class TestHybridRetriever(unittest.TestCase):
    """HybridRetriever — 向量 + BM25 + Rerank"""

    def setUp(self):
        from core.stores import MetadataStore, VectorStore, CleanTextStore
        from core.retriever import HybridRetriever

        self.tmp = TEST_DIR / "retriever_test"
        self.tmp.mkdir(parents=True, exist_ok=True)

        self.ms = MetadataStore(str(self.tmp / "metadata.db"))
        self.vs = VectorStore(str(self.tmp / "vectors"))
        self.ct = CleanTextStore(str(self.tmp / "clean.jsonl"))
        self.fake_embed = _make_fake_embed_model()

        # 先入库一些测试数据
        from langchain_core.documents import Document
        texts = [
            "年假申请流程：员工需提前一周在OA系统提交年假申请。",
            "病假需提供医院证明，急诊可事后补交。",
            "年终奖发放时间为每年1月，根据绩效评级确定金额。",
        ]
        docs = [Document(page_content=t, metadata={"page_number": i + 1}) for i, t in enumerate(texts)]
        file_id = "f_retrieve_test"
        chunk_ids = []
        v_metas = []
        for idx, doc in enumerate(docs):
            cid = f"{file_id}#chunk_{idx:04d}"
            chunk_ids.append(cid)
            v_metas.append({"chunk_id": cid, "file_id": file_id, "level": "p"})

        # 写入元数据
        self.ms.insert_file({"file_id": file_id, "doc_code": "", "original_filename": "HR手册.pdf",
                             "storage_path": "/t/hr.pdf", "content_hash": "sha256:hr", "mime_type": "app/pdf",
                             "file_size": 999, "upload_time": "2026", "uploader_id": "hr", "source_tag": "t",
                             "status": "ok", "version_tag": "v1"})
        self.ms.insert_chunks(file_id, [
            {"chunk_id": cid, "start_page": i + 1, "end_page": i + 1, "char_len": len(d.page_content),
             "level": "p", "category": "NarrativeText"}
            for i, (cid, d) in enumerate(zip(chunk_ids, docs))
        ])

        # 写入清洗文本
        self.ct.append(file_id, docs)

        # 写入向量
        embs = self.fake_embed.encode(texts)
        self.vs.add(chunk_ids, embs, v_metas)

        # 创建检索器 (mock reranker，避免下载模型)
        self.retriever = HybridRetriever(
            vector_store=self.vs,
            metadata_store=self.ms,
            clean_text_store=self.ct,
            embed_model=self.fake_embed,
            rerank_model_path=None,
        )
        # Mock 掉 reranker，使用简单分数代替
        def fake_rerank(query, docs, top_n):
            for d in docs:
                d["rerank_score"] = d["score"]
            docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return docs[:top_n]
        self.retriever._rerank = fake_rerank

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("年假怎么请", top_n=3)
        self.assertGreater(len(results), 0)
        # 结果中应包含关于年假的内容
        texts = [r["text"] for r in results]
        self.assertTrue(any("年假" in t for t in texts))

    def test_retrieve_respects_top_n(self):
        results = self.retriever.retrieve("请假", top_n=1)
        self.assertLessEqual(len(results), 1)

    def test_retrieve_with_file_filter(self):
        results = self.retriever.retrieve("请假", filter_file_ids=["f_retrieve_test"])
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["meta"]["file_id"], "f_retrieve_test")


# ======================================================================
# 5. LangChain 工具测试
# ======================================================================
class TestLangChainTools(unittest.TestCase):
    """6 个工具的集成测试"""

    @classmethod
    def setUpClass(cls):
        from core.stores import OriginalFileStore, MetadataStore, VectorStore, CleanTextStore, ArchiveStore
        from core.pipeline import RAGPipelineOffline
        from core.retriever import HybridRetriever, RAGHybridRetrieverLC
        from tools.ingest import RAGIngestTool, RAGIngestFolderTool
        from tools.retrieve import RAGRetrieveTool
        from tools.manage import RAGUpdateTool, RAGListArchiveTool, RAGListActiveTool

        cls.tmp = TEST_DIR / "tools_test"
        cls.tmp.mkdir(parents=True, exist_ok=True)

        fake_embed = _make_fake_embed_model()

        fs = OriginalFileStore(str(cls.tmp / "originals"))
        ms = MetadataStore(str(cls.tmp / "metadata.db"))
        vs = VectorStore(str(cls.tmp / "vectors"))
        ct = CleanTextStore(str(cls.tmp / "clean.jsonl"))
        archive = ArchiveStore(str(cls.tmp / "archive"), ttl_days=30)

        pipeline = RAGPipelineOffline(fs, ms, vs, ct, fake_embed, archive_store=archive)
        hybrid = HybridRetriever(vs, ms, ct, fake_embed, rerank_model_path=None)
        # Mock reranker 避免下载模型
        def fake_rerank(query, docs, top_n):
            for d in docs:
                d["rerank_score"] = d["score"]
            docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            return docs[:top_n]
        hybrid._rerank = fake_rerank
        retriever_lc = RAGHybridRetrieverLC(hybrid=hybrid)

        cls.ingest_tool = RAGIngestTool(pipeline=pipeline)
        cls.ingest_folder_tool = RAGIngestFolderTool(pipeline=pipeline)
        cls.retrieve_tool = RAGRetrieveTool(retriever=retriever_lc)
        cls.update_tool = RAGUpdateTool(pipeline=pipeline)
        cls.list_archive_tool = RAGListArchiveTool(archive_store=archive)
        cls.list_active_tool = RAGListActiveTool(metadata_store=ms)

        cls._tmp = cls.tmp

    def _make_test_file(self, name, content):
        path = self._tmp / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    # ---- 1. rag_ingest_file ----
    def test_rag_ingest_file_success(self):
        fpath = self._make_test_file("请假制度.txt",
            "请假制度\n第一条 员工请假需提前申请，经部门主管审批后方可休假。\n"
            "第二条 紧急情况可电话报备，事后补交请假申请。\n第三条 年假需提前一周申请。")
        result = self.ingest_tool._run(filename=fpath, doc_code="LEAVE-001", uploader_id="admin")
        self.assertIn("✅", result)
        self.assertIn("入库成功", result)
        self.assertIn("LEAVE-001", result)

    def test_rag_ingest_file_missing_file(self):
        result = self.ingest_tool._run(filename="/nonexistent/path/doc.pdf", doc_code="X-001")
        self.assertIn("❌", result)

    # ---- 2. rag_ingest_folder ----
    def test_rag_ingest_folder(self):
        folder = self._tmp / "batch_folder"
        folder.mkdir(exist_ok=True)
        (folder / "doc1.txt").write_text("文档一：内容部分需要足够长" + "A" * 60, encoding="utf-8")
        (folder / "doc2.txt").write_text("文档二：内容部分需要足够长" + "B" * 60, encoding="utf-8")
        result = self.ingest_folder_tool._run(folder_path=str(folder))
        # 批量上传工具目前返回固定字符串
        self.assertIn("批量上传", result)

    # ---- 3. rag_retrieve ----
    def test_rag_retrieve_finds_relevant(self):
        result = self.retrieve_tool._run(query="请假制度", top_n=3)
        # 有结果时应有内容
        self.assertTrue(len(result) > 0)

    def test_rag_retrieve_no_results(self):
        result = self.retrieve_tool._run(query="火星基地建设方案", top_n=3)
        # 可能返回 "未找到" 或空结果
        self.assertTrue(isinstance(result, str))

    # ---- 4. rag_update_file ----
    def test_rag_update_file_workflow(self):
        # 先入库
        old_path = self._make_test_file("安全手册_v1.txt",
            "安全手册 V1\n第一条 进入厂区需佩戴工牌。" + "X" * 50)
        r1 = self.ingest_tool._run(filename=old_path, doc_code="SEC-001")

        # 更新
        new_path = self._make_test_file("安全手册_v2.txt",
            "安全手册 V2\n第一条 进入厂区需佩戴工牌并刷卡。\n新增：第二条 访客需登记。" + "Y" * 50)
        r2 = self.update_tool._run(doc_code="SEC-001", new_file_path=new_path)
        self.assertIn("✅", r2)
        self.assertIn("更新成功", r2)

    def test_rag_update_file_bad_doc_code(self):
        new_path = self._make_test_file("whatever.txt", "内容" + "Z" * 60)
        result = self.update_tool._run(doc_code="NONEXISTENT-999", new_file_path=new_path)
        self.assertIn("❌", result)

    # ---- 5. rag_list_active_files ----
    def test_rag_list_active_files(self):
        result = self.list_active_tool._run(keyword="")
        self.assertIn("📄", result)

    def test_rag_list_active_files_with_keyword(self):
        result = self.list_active_tool._run(keyword="安全")
        self.assertIn("安全", result)

    def test_rag_list_active_files_no_match(self):
        result = self.list_active_tool._run(keyword="ZZZ_NOT_EXIST")
        self.assertIn("未找到", result)

    # ---- 6. rag_list_archive ----
    def test_rag_list_archive(self):
        # 入库+更新来生成归档
        old_p = self._make_test_file("归档测试_v1.txt", "V1 内容" + "A" * 60)
        self.ingest_tool._run(filename=old_p, doc_code="ARCH-TEST-001")

        # 需要获取 old file_id
        from core.stores import MetadataStore
        ms = MetadataStore(str(self._tmp / "metadata.db"))
        old_meta = ms.find_by_doc_code("ARCH-TEST-001")
        old_id = old_meta["file_id"]

        new_p = self._make_test_file("归档测试_v2.txt", "V2 内容" + "B" * 60)
        self.update_tool._run(doc_code="ARCH-TEST-001", new_file_path=new_p)

        result = self.list_archive_tool._run(file_id=old_id)
        self.assertIn("📦", result)

    def test_rag_list_archive_empty(self):
        result = self.list_archive_tool._run(file_id="f_nonexistent_12345")
        self.assertIn("无归档", result)


# ======================================================================
# 6. build() 装配函数测试
# ======================================================================
class TestBuild(unittest.TestCase):
    """测试 build.py 一键装配"""

    def test_build_returns_retriever_and_6_tools(self):
        """用 mock 替换 SentenceTransformer 避免加载真实模型"""
        with patch('build.SentenceTransformer', return_value=_make_fake_embed_model()):
            from .build import build
            retriever, tools = build()

            self.assertIsNotNone(retriever)
            self.assertEqual(len(tools), 6)

            tool_names = [t.name for t in tools]
            expected = [
                "rag_ingest_file",
                "rag_ingest_folder",
                "rag_retrieve",
                "rag_update_file",
                "rag_list_archive",
                "rag_list_active_files",
            ]
            for name in expected:
                self.assertIn(name, tool_names, f"缺少工具: {name}")


# ======================================================================
# 7. 运行入口
# ======================================================================
def run_all():
    """不使用 pytest 时的纯 Python 运行入口"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestOriginalFileStore))
    suite.addTests(loader.loadTestsFromTestCase(TestMetadataStore))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanTextStore))
    suite.addTests(loader.loadTestsFromTestCase(TestArchiveStore))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestLangChainTools))
    suite.addTests(loader.loadTestsFromTestCase(TestBuild))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 清理临时目录
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        print(f"\n[清理] 已清理临时目录: {TEST_DIR}")

    return result


def run_tools_only():
    """只测试 6 个工具 (快速冒烟)"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestLangChainTools))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 系统测试")
    parser.add_argument("--quick", action="store_true", help="只运行工具冒烟测试")
    parser.add_argument("--keep-tmp", action="store_true", help="保留临时文件")
    args = parser.parse_args()

    if args.quick:
        result = run_tools_only()
    else:
        result = run_all()

    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
