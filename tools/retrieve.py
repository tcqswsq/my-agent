from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever

class RetrieveInput(BaseModel):
    query: str = Field(description="用户的自然语言问题")
    top_n: int = Field(default=5)

class RAGRetrieveTool(BaseTool):
    name: str = "rag_retrieve"
    description: str = "🔍 从企业知识库中检索相关文档片段来回答问题。"
    args_schema: type = RetrieveInput
    retriever: BaseRetriever = None

    def _run(self, query: str, top_n: int = 5) -> str:
        docs = self.retriever.invoke(query)
        if not docs: return "知识库中未找到相关内容。"
        results = [f"[{i}] 来源《{d.metadata.get('source','未知')}》：{d.page_content.strip().replace(chr(10),' ')[:200]}..." for i,d in enumerate(docs[:top_n],1)]
        return "\n\n".join(results)