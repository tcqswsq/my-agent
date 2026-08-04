import os
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path


class UpdateFileInput(BaseModel):
    doc_code: str = Field(description="要更新的文件业务编码（如 EMP-HB-001）")
    new_file_path: str = Field(description="新文件的本地路径")

class RAGUpdateTool(BaseTool):
    name: str = "rag_update_file"
    description: str = "🔄 更新知识库文件。用 doc_code 定位旧文件，自动归档旧版并入库新版。"
    args_schema: type = UpdateFileInput
    pipeline: object = None

    def _run(self, doc_code: str, new_file_path: str) -> str:
        try:
            with open(new_file_path, "rb") as f:
                stream = f.read()
            result = self.pipeline.update_file(doc_code, stream, Path(new_file_path).name)
            return f"✅ 更新成功：doc_code={doc_code}，新 file_id={result['file_id']}"
        except Exception as e:
            return f"❌ 更新失败：{str(e)}"

class ListArchiveInput(BaseModel):
    file_id: str = Field(description="文件ID")

class RAGListArchiveTool(BaseTool):
    name: str = "rag_list_archive"
    description: str = "📦 查看某个文件的历史归档版本。"
    args_schema: type = ListArchiveInput
    archive_store: object = None

    def _run(self, file_id: str) -> str:
        versions = self.archive_store.list_versions(file_id)
        return "\n".join(versions) if versions else "无归档版本"
class ListActiveInput(BaseModel):
    keyword: str = Field(default="", description="可选，按文件名关键词过滤")
    category: str = Field(default="", description="可选，按分类过滤：制度规章/技术文档/财务资料/人力资源/项目管理/产品文档/培训材料/其他")
    tag: str = Field(default="", description="可选，按标签过滤")

class RAGListActiveTool(BaseTool):
    name: str = "rag_list_active_files"
    description: str = "📋 列出知识库中所有活跃文件（名称、doc_code、file_id、分类、标签）。可按关键词、分类、标签筛选。"
    args_schema: type = ListActiveInput
    metadata_store: object = None

    def _run(self, keyword: str = "", category: str = "", tag: str = "") -> str:
        files = self.metadata_store.get_active_files(category=category, tag=tag)
        if not files:
            return "知识库为空，没有任何活跃文件。"
        lines = []
        for f in files:
            if keyword and keyword not in f["original_filename"]:
                continue
            extra = f" | 分类={f.get('category', '其他')}"
            if f.get('tags'):
                extra += f" | 标签={f['tags']}"
            lines.append(f"📄 {f['original_filename']} | doc_code={f['doc_code']} | id={f['file_id']}{extra}")
        return "\n".join(lines) if lines else f"未找到含 '{keyword}' 的文件。"


class SetMetadataInput(BaseModel):
    file_id: str = Field(description="要修改的文件 file_id")
    category: str = Field(default="", description="新的分类：制度规章/技术文档/财务资料/人力资源/项目管理/产品文档/培训材料/其他")
    tags: str = Field(default="", description="新的标签，多个标签用逗号分隔，如 '考勤,年假,2026版'")


class RAGSetMetadataTool(BaseTool):
    name: str = "rag_set_metadata"
    description: str = "🏷️ 修改文件的分类和标签。可以单独改分类或标签，也可以一起改。"
    args_schema: type = SetMetadataInput
    metadata_store: object = None

    def _run(self, file_id: str, category: str = "", tags: str = "") -> str:
        ok = self.metadata_store.update_file_metadata(file_id, category=category, tags=tags)
        if ok:
            return f"✅ 已更新 {file_id} 的元数据：分类={category or '不变'}，标签={tags or '不变'}"
        return "❌ 请至少指定 category 或 tags 中的一个。"


class DeleteFileInput(BaseModel):
    doc_code: str = Field(default="", description="要删除文件的业务编码（与 file_id 二选一）")
    file_id: str = Field(default="", description="要删除文件的 file_id（与 doc_code 二选一）")


class RAGDeleteTool(BaseTool):
    name: str = "rag_delete_file"
    description: str = "🗑️ 永久删除知识库文件。通过 doc_code 或 file_id 定位，停用元数据、删除向量索引、删除全文索引、从磁盘移除原始文件。⚠️ 此操作不可逆！"
    args_schema: type = DeleteFileInput
    metadata_store: object = None
    vector_store: object = None
    clean_text_store: object = None

    def _run(self, doc_code: str = "", file_id: str = "") -> str:
        # 通过 doc_code 或 file_id 定位文件
        if doc_code and not file_id:
            meta = self.metadata_store.find_by_doc_code(doc_code)
            if not meta:
                return f"❌ 未找到 doc_code={doc_code} 的活跃文件，可能已被删除或不存在。"
            file_id = meta["file_id"]
            storage_path = meta.get("storage_path", "")
            original_name = meta.get("original_filename", "")
        elif file_id:
            meta = self.metadata_store.get_file(file_id)
            if not meta:
                return f"❌ 未找到 file_id={file_id} 的文件。"
            storage_path = meta.get("storage_path", "")
            original_name = meta.get("original_filename", "")
        else:
            return "❌ 请提供 doc_code 或 file_id 来定位要删除的文件。"

        # 1. 停用元数据
        self.metadata_store.deactivate_file(file_id)
        # 2. 删除向量索引
        self.vector_store.delete_by_file_id(file_id)
        # 3. 删除全文索引（FTS5）
        if self.clean_text_store:
            self.clean_text_store.delete_by_file_id(file_id)
        # 4. 从磁盘删除原始文件
        if storage_path:
            sp = Path(storage_path)
            if sp.exists():
                sp.unlink()

        return f"✅ 已永久删除文件「{original_name}」：file_id={file_id}"