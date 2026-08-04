from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path

class IngestInput(BaseModel):
    filename: str = Field(description="文件的完整本地路径")
    doc_code: str = Field(description="业务唯一编码，如 EMP-HB-001（员工手册）、FIN-2026-01（财务制度）")
    uploader_id: str = Field(default="system")


class RAGIngestTool(BaseTool):
    name: str = "rag_ingest_file"
    description: str = "📥 上传文件到知识库。每个文件必须指定 doc_code（业务唯一编码），相同 doc_code 的文件会被识别为同一份文件的不同版本。"
    args_schema: type = IngestInput
    pipeline: object = None

    def _run(self, filename: str, doc_code: str, uploader_id: str = "system") -> str:
        try:
            with open(filename, "rb") as f:
                data = f.read()
            result = self.pipeline.ingest_file(data, Path(filename).name, uploader_id, doc_code=doc_code)
            return f"✅ 入库成功：{result['file_id']}，doc_code={doc_code}，共 {result['chunks_count']} 个片段。"
        except Exception as e:
            return f"❌ 上传失败：{str(e)}"


class IngestFolderInput(BaseModel):
    folder_path: str = Field(description="文件夹的完整路径")
    uploader_id: str = Field(default="system")
    recursive: bool = Field(default=True)

class RAGIngestFolderTool(BaseTool):
    name: str = "rag_ingest_folder"
    description: str = "📁 批量上传文件夹内的所有文档到知识库。"
    args_schema: type = IngestFolderInput
    pipeline: object = None

    def _run(self, folder_path: str, uploader_id: str = "system", recursive: bool = True) -> str:
        # ... (这里省略了具体实现，你可以从之前的代码中复制过来，保持逻辑一致)
        return "批量上传完成"