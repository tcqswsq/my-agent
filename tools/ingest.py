from typing import ClassVar
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path

class IngestInput(BaseModel):
    filename: str = Field(description="文件的完整本地路径")
    doc_code: str = Field(description="业务唯一编码，如 EMP-HB-001（员工手册）、FIN-2026-01（财务制度）")
    uploader_id: str = Field(default="system")
    category: str = Field(default="其他", description="文档分类：制度规章/技术文档/财务资料/人力资源/项目管理/产品文档/培训材料/其他")
    tags: str = Field(default="", description="标签，多个标签用逗号分隔，如 '考勤,年假,2026版'")


class RAGIngestTool(BaseTool):
    name: str = "rag_ingest_file"
    description: str = "📥 上传文件到知识库。每个文件必须指定 doc_code（业务唯一编码），相同 doc_code 的文件会被识别为同一份文件的不同版本。可选指定分类(category)和标签(tags)。"
    args_schema: type = IngestInput
    pipeline: object = None

    def _run(self, filename: str, doc_code: str, uploader_id: str = "system",
             category: str = "其他", tags: str = "") -> str:
        try:
            with open(filename, "rb") as f:
                data = f.read()
            result = self.pipeline.ingest_file(data, Path(filename).name, uploader_id,
                                               doc_code=doc_code, category=category, tags=tags)
            return f"✅ 入库成功：{result['file_id']}，doc_code={doc_code}，分类={category}，共 {result['chunks_count']} 个片段。"
        except Exception as e:
            return f"❌ 上传失败：{str(e)}"


class IngestFolderInput(BaseModel):
    folder_path: str = Field(description="文件夹的完整路径")
    uploader_id: str = Field(default="system")
    recursive: bool = Field(default=True)

class RAGIngestFolderTool(BaseTool):
    name: str = "rag_ingest_folder"
    description: str = "📁 批量上传文件夹内的所有文档到知识库。支持递归子目录。"
    args_schema: type = IngestFolderInput
    pipeline: object = None

    # 支持的文件后缀
    SUPPORTED_EXT: ClassVar[set] = {'.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls',
                                     '.csv', '.pptx', '.ppt', '.html', '.htm', '.json'}

    def _run(self, folder_path: str, uploader_id: str = "system", recursive: bool = True) -> str:
        path = Path(folder_path)
        if not path.exists():
            return f"❌ 文件夹不存在: {folder_path}"
        if not path.is_dir():
            return f"❌ 路径不是文件夹: {folder_path}"

        # 扫描文件
        if recursive:
            files = [f for f in path.rglob('*') if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXT]
        else:
            files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXT]

        if not files:
            return f"⚠️ 文件夹中没有支持的文件类型: {folder_path}\n支持: {', '.join(sorted(self.SUPPORTED_EXT))}"

        results = []
        success_count = 0
        for fp in files:
            try:
                with open(fp, "rb") as fh:
                    data = fh.read()
                # 用文件名（不含扩展名）作为 doc_code
                doc_code = fp.stem
                result = self.pipeline.ingest_file(data, fp.name, uploader_id, doc_code=doc_code)
                results.append(f"  ✅ {fp.name} → {result['chunks_count']} chunks (id={result['file_id']})")
                success_count += 1
            except Exception as e:
                results.append(f"  ❌ {fp.name}: {str(e)[:100]}")

        summary = f"📁 批量上传完成: {success_count}/{len(files)} 个文件成功"
        return summary + "\n" + "\n".join(results)