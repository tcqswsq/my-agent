import re
import json

from langchainq.peizhi import model
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_openai import ChatOpenAI
from langchainq.tool.rag import search
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from langchainq.tool.document_lifecycle import (
    index_documents, update_documents, delete_documents,
    cleanup_knowledge_base, get_kb_status_tool
)
from langchain_huggingface import HuggingFaceEmbeddings
from rag_system.build import build
import os
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(
 model=os.getenv("MODEL"),
 openai_api_key=os.getenv("API"),
 openai_api_base=os.getenv("URL"),
 temperature = 0.7,
)
# ========== 1. 配置区（按建议设定，集中管理）==========
VECTOR_THRESHOLD = 0.65
BM25_THRESHOLD = 1.0
RERANK_TOP_K = 5
RELEVANCE_THRESHOLD = 0.5
COMPLIANCE_KEYWORDS = [] # 业务自行填充
# ========== 2. 数据结构 ==========
class Chunk:
    def __init__(self, id: str, text: str, vector_score: float, bm25_score: float, metadata: Optional[Dict] = None):
        self.id = id
        self.text = text
        self.vector_score = vector_score
        self.bm25_score = bm25_score
        self.metadata = metadata or {}
        self.rerank_score = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """从工具返回的 dict 转成 Chunk"""
        return cls(
            id=data["id"],
            text=data["text"],
            vector_score=data["vector_score"],
            bm25_score=data["bm25_score"],
            metadata=data.get("metadata", {}),
        )
class State(MessagesState):
    raw_chunks: Optional[List[Chunk]] = None
    filtered_chunks: Optional[List[Chunk]] = None
    reranked_chunks: Optional[List[Chunk]] = None
    context: Optional[str] = None
    tool_call_sequence: List[str] = []

    # ✅ 2. 本轮对话工具调用总次数（用于限制连续调用）
    tool_call_count: int = 0
    final_answer: Optional[str] = None
    confidence: Optional[str] = None
    validation_warnings: List[str] = []
def step1_filter(state: State) -> dict:

    """阈值过滤：向量 + BM25 双路 AND"""
    chunks = state.get("raw_chunks") or []
    filtered = [
        c for c in chunks
        if c.vector_score >= VECTOR_THRESHOLD and c.bm25_score >= BM25_THRESHOLD
    ]
    return {"filtered_chunks": filtered}
def step2_rerank(state: State) -> dict:
    """Rerank 精排"""
    chunks = state.get("filtered_chunks") or []
    for c in chunks:
        c.rerank_score = c.vector_score * 0.7 + min(c.bm25_score / 10, 1) * 0.3
    valid = [c for c in chunks if c.rerank_score >= RELEVANCE_THRESHOLD]
    reranked = sorted(valid, key=lambda x: x.rerank_score, reverse=True)[:RERANK_TOP_K]
    return {"reranked_chunks": reranked}


def step3_assemble(state: State) -> dict:
    """上下文组装"""
    chunks = state.get("reranked_chunks") or []

    if not chunks:
        # 明确告诉模型：工具调用过了，但是空的
        context = "系统提示：知识库中未检索到与用户问题相关的文档内容。"
    else:
        context = "\n\n---\n\n".join(
            [f"[{c.id}] {c.metadata} {c.text}" for c in chunks]
        )
    return {"context": context}
retriever,tool_1 = build()
# index_documents, update_documents, delete_documents,cleanup_knowledge_base, get_kb_status_tool,ingest_tool, retrieve_tool
all_tools = [
    *tool_1
]
_tools = model.bind_tools(all_tools)
CURRENT_DATE = datetime.now().strftime("%Y年%m月%d日")
SYSTEM_PROMPT =SYSTEM_PROMPT = f"""你是企业内部专属智能助手，无法访问互联网。当前日期：{CURRENT_DATE}。

### 行为准则
1. **反幻觉**：关于企业内部的信息（制度、人员、项目、数据），100%基于工具返回结果，严禁编造。
2. **工具区分**：
   - 查内部资料 -> 调用 `rag_retrieve`。
   - 上传内部文档 -> 调用 `rag_ingest_file`。
   - 通用知识（代码、数学、公开常识）-> 直接回答。
3. **空结果处理**：若检索无结果，必须告知用户“知识库中未找到相关内容，建议上传相关文档”。
4. **前置校验**：调用 `rag_ingest_file` 前必须确认用户提供了文件路径，否则先询问路径。
### 文件更新规则
当用户提出更新文件的请求时（如“把员工手册更新到2026版”）：
1. 必须先调用`rag_retrieve`工具，用文件的业务名称（如“员工手册”）检索，获取对应活跃文件的`file_id`
2. 再调用`rag_update_file`工具，传入查询到的`old_file_id`和新文件路径
3. 禁止在没有`old_file_id`的情况下直接上传新文件（避免产生重复文件）

### 输出格式（严格遵守）
---
思考: [问题类型判断 + 工具选择理由]
最终答案: [无需工具时填写答案；需要工具时必须留空]
---
### 示例
用户：2025年研发项目代号有哪些？
---
思考: 涉及企业内部项目信息，需调用 `rag_retrieve` 确认，不可臆测。
最终答案: 
---
用户：帮我上传 D:\手册.pdf
---
思考: 用户提供了明确的文件路径，需调用 `rag_ingest_file` 进行入库。
最终答案: 
---
用户：你好

思考: 通用问候，无需调用工具，直接回复。
最终答案: 您好，我是您的专属智能助手，请问有什么可以帮您？
"""
def agent_node(state: State) -> dict:
    messages = state.get("messages", [])
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    # 如果有 RAG 上下文，追加到 messages
    if state.get("context"):
        messages = messages + [
            HumanMessage(content=f"以下是参考资料，请基于这些内容回答：\n\n{state['context']}")
        ]
    response = _tools.invoke(messages)

    # ✅ 关键：记录工具调用
    updates = {"messages": [response]}
    # 如果这次回复包含了工具调用
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # 1. 更新调用序列
        sequence = state.get("tool_call_sequence", [])
        for tc in response.tool_calls:
            sequence.append(tc["name"])
        # 2. 更新调用次数
        count = state.get("tool_call_count", 0) + len(response.tool_calls)
        updates["tool_call_sequence"] = sequence
        updates["tool_call_count"] = count

        print(f"\n🔧 [工具调用] 序列: {sequence}, 本轮累计: {count}")

    return updates
def parse_tool_result(state: State) -> dict:
    tool_messages = [m for m in state.get("messages", []) if isinstance(m, ToolMessage)]
    if not tool_messages:
        return {"raw_chunks": []}

    last = tool_messages[-1]
    if last.name != "search":
        return {"raw_chunks": []}

    try:
        raw_data = json.loads(last.content)
        chunks = [Chunk.from_dict(item) for item in raw_data]
        return {"raw_chunks": chunks}
    except Exception:
        return {"raw_chunks": []}
#最后检测
def step5_validater(state: State) -> dict:
    warnings = []
    answer = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    conf = "中"
    valid_ids = {c.id for c in (state.get("reranked_chunks") or [])}

    cited = set(re.findall(r"\[(.+?)\]", answer))
    bad_refs = cited - valid_ids
    if bad_refs:
        warnings.append(f"无效引用: {bad_refs}")
        conf = "低"

    nums = set(re.findall(r"\d+", answer))
    if nums:
        src = " ".join(c.text for c in (state.get("reranked_chunks") or []))
        missing = [n for n in nums if n not in src]
        if missing:
            warnings.append(f"数字幻觉: {missing}")
            conf = "低"

    for kw in COMPLIANCE_KEYWORDS:
        if kw in answer:
            warnings.append(f"合规风险: {kw}")
            conf = "低"
    print("\n🟢 [NODE] 已检验合格")
    return {
        "final_answer": answer,
        "confidence": conf,
        "validation_warnings": warnings,
    }
#========设置工具最大循环次数=========
MAX_TOOL_CALLS_PER_TURN = 7
def should_continue(state: State) -> str:
    """双重熔断检查"""
    last_msg = state["messages"][-1]

    # 如果没有工具调用，直接去验证
    if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
        return "validate"

    # --- 熔断规则 1：连续调用次数限制 ---
    tool_call_count = state.get("tool_call_count", 0)
    if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
        print(f"\n🚨 [熔断] 达到最大工具调用次数 ({MAX_TOOL_CALLS_PER_TURN})，强制结束。")
        return "validate"

    # --- 熔断规则 2：震荡死循环检测 ---
    seq = state.get("tool_call_sequence", [])

    # 我们只检测最近4次调用：A -> B -> A -> B
    if len(seq) >= 4:
        # 取最后4个
        a, b, c, d = seq[-4:]

        # 判断是否为震荡模式：A != B 且 A == C 且 B == D
        # 示例：search -> crawl -> search -> crawl
        if a != b and a == c and b == d:
            print(f"\n🚨 [熔断] 检测到震荡死循环: {a} -> {b} -> {c} -> {d}，强制结束。")
            return "validate"

    # 一切正常，允许调用工具
    return "tools"
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

def trim_messages_node(state: State) -> dict:
    """
    1. 裁剪消息，防止 Token 爆炸
    2. 重置工具调用状态（关键：防止跨轮死循环）
    """
    messages = state.get("messages", [])

    # ---------- 1. 裁剪消息（最简单、最稳的方案）----------
    # 保留 SystemMessage + 最近 N 条消息
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_sys_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    # ✅ 只保留最近 10 条非系统消息（可根据模型调整）
    MAX_HISTORY = 10
    trimmed_non_sys = non_sys_msgs[-MAX_HISTORY:]

    trimmed_messages = system_msgs + trimmed_non_sys

    # ---------- 2. 重置工具调用状态（核心）----------
    # 判断是否是“新的一轮用户提问”
    # 规则：如果裁剪后最后一条是 HumanMessage，说明用户刚发新问题
    if trimmed_messages and isinstance(trimmed_messages[-1], HumanMessage):
        print("\n🔄 [Reset] 检测到新用户提问，重置工具调用状态")
        return {
            "messages": trimmed_messages,
            "tool_call_sequence": [],
            "tool_call_count": 0
        }

    # 同一轮对话，不清零
    return {"messages": trimmed_messages}
builder = StateGraph(State)

# 节点
builder.add_node("trim_messages", trim_messages_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("parse_tool_result", parse_tool_result)
builder.add_node("step1_filter", step1_filter)
builder.add_node("step2_rerank", step2_rerank)
builder.add_node("step3_assemble", step3_assemble)
builder.add_node("validate", step5_validater)

# ========== 正确的边 ==========

# 1. 入口：先 trim（重置状态），再进 agent
builder.add_edge(START, "trim_messages")
builder.add_edge("trim_messages", "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",      # 有工具调用 → 执行工具
        "validate": "validate" # 无工具调用 → 验证结束
    },
)
# 3. 工具执行链路（一条直线，中间不 trim）
builder.add_edge("tools", "parse_tool_result")
builder.add_edge("parse_tool_result", "step1_filter")
builder.add_edge("step1_filter", "step2_rerank")
builder.add_edge("step2_rerank", "step3_assemble")


builder.add_edge("step3_assemble", "agent")

# 5. 验证后结束
builder.add_edge("validate", END)

# 编译
graph = builder.compile(checkpointer=InMemorySaver())
# ====================== 主函数 ======================
def run():
    try:
        while True:
            config = {"configurable": {"thread_id": "会话1", "user_id": "user_123"}}
            user_input = input("[会话1] 你: ")
            if user_input.lower() == "exit":
                print("\n👋 正常退出，再见！")
                break
            output = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
            print(f"AI: {output['messages'][-1].content}\n")
    except KeyboardInterrupt:
        # 专门处理Ctrl+C中断，不打冗长traceback
        print("\n👋 手动中断，退出对话，再见！")
    except Exception as e:

        print(f"\n❌ 运行异常: {e}")
        raise  # 重新抛出，保留完整调试信息

if __name__ == "__main__":
    # print(graph.get_graph().draw_mermaid())
    run()