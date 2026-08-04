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

class RAGListActiveTool(BaseTool):
    name: str = "rag_list_active_files"
    description: str = "📋 列出知识库中所有活跃文件（名称、doc_code、file_id），用于确认要更新的目标。"
    args_schema: type = ListActiveInput
    metadata_store: object = None

    def _run(self, keyword: str = "") -> str:
        files = self.metadata_store.get_active_files()
        if not files:
            return "知识库为空，没有任何活跃文件。"
        lines = []
        for f in files:
            if keyword and keyword not in f["original_filename"]:
                continue
            lines.append(f"📄 {f['original_filename']} | doc_code={f['doc_code']} | id={f['file_id']}")
        return "\n".join(lines) if lines else f"未找到含 '{keyword}' 的文件。"