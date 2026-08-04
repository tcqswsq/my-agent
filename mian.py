import re
import json

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode
from rag_system.build import build
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model=os.getenv("MODEL"),
    openai_api_key=os.getenv("API"),
    openai_api_base=os.getenv("URL"),
    temperature=0.7,
)

# ========== 配置 ==========
COMPLIANCE_KEYWORDS = []  # 业务自行填充

# ========== 状态 ==========
class State(MessagesState):
    tool_call_sequence: List[str] = []
    tool_call_count: int = 0
    final_answer: Optional[str] = None
    confidence: Optional[str] = None
    validation_warnings: List[str] = []


retriever, tool_1 = build()
all_tools = list(tool_1)
_tools = model.bind_tools(all_tools)

CURRENT_DATE = datetime.now().strftime("%Y年%m月%d日")
SYSTEM_PROMPT = f"""你是企业内部专属智能助手，可调用本地知识库工具和外部 MCP 服务。当前日期：{CURRENT_DATE}。

### 可用工具
系统提供的工具都有名称(name)和描述(description)，大模型会自动看到。选工具时看 description，传参时看参数说明。
- 知识库工具：检索、上传、更新、删除、列出、归档内部文档
- 外部 MCP 工具：GitHub、网页搜索等第三方服务

### 行为准则
1. **反幻觉**：涉及企业内部信息，必须基于工具返回结果，严禁编造。
2. **检索优先**：涉及内部制度、规章、项目信息，先检索知识库再回答。
3. **空结果处理**：知识库检索无结果时，告知用户并建议上传相关文档。
4. **上传前确认**：上传文件前确认用户提供了文件路径和 doc_code。
5. **更新流程**：先检索确认文件存在，用 doc_code 更新（系统自动归档旧版）。
6. **外部能力**：知识库无法回答时，可调用 MCP 外部工具辅助。
7. **通用知识**：代码、数学、公开常识直接回答，无需调工具。
### 输出格式
---
思考: [判断用户意图，选择合适的工具或直接回答]
最终答案: [需要调工具时留空，直接回答时填写]
---
"""


def agent_node(state: State) -> dict:
    """Agent 节点：调用 LLM，LLM 决定是回答还是调工具"""
    messages = state.get("messages", [])
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = _tools.invoke(messages)

    updates = {"messages": [response]}
    if hasattr(response, 'tool_calls') and response.tool_calls:
        sequence = state.get("tool_call_sequence", [])
        for tc in response.tool_calls:
            sequence.append(tc["name"])
        count = state.get("tool_call_count", 0) + len(response.tool_calls)
        updates["tool_call_sequence"] = sequence
        updates["tool_call_count"] = count
        print(f"\n🔧 [工具调用] 序列: {sequence}, 本轮累计: {count}")

    return updates


def validate(state: State) -> dict:
    """最终验证：检查引用、数字幻觉、合规风险"""
    warnings = []
    answer = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content
            break

    conf = "中"

    # 检查引用
    cited = set(re.findall(r"\[(.+?)\]", answer))
    if cited:
        # 从工具返回中提取已知来源文件名
        known_sources = set()
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                known_sources.update(re.findall(r"《(.+?)》", str(msg.content)))
        bad_refs = cited - known_sources
        if bad_refs and known_sources:
            warnings.append(f"引用存疑: {bad_refs}")
            conf = "低"

    # 检查数字幻觉
    nums = set(re.findall(r"\d+", answer))
    if nums and len(answer) > 50:
        src = ""
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                src += str(msg.content) + " "
        missing = [n for n in nums if n not in src]
        if missing and len(missing) > len(nums) * 0.5:
            warnings.append(f"数字存疑: {missing[:5]}...")
            conf = "低"

    # 合规关键词扫描
    for kw in COMPLIANCE_KEYWORDS:
        if kw in answer:
            warnings.append(f"合规风险: {kw}")
            conf = "低"

    print(f"\n🟢 [NODE] 验证完成，置信度: {conf}")
    return {
        "final_answer": answer,
        "confidence": conf,
        "validation_warnings": warnings,
    }


# ========== 熔断控制 ==========
MAX_TOOL_CALLS_PER_TURN = 7


def should_continue(state: State) -> str:
    """检查是否继续调工具 → 还是去验证"""
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
        return "validate"

    # 熔断 1：最大调用次数
    tool_call_count = state.get("tool_call_count", 0)
    if tool_call_count >= MAX_TOOL_CALLS_PER_TURN:
        print(f"\n🚨 [熔断] 达到最大工具调用次数 ({MAX_TOOL_CALLS_PER_TURN})，强制结束。")
        return "validate"

    # 熔断 2：震荡死循环检测 (A→B→A→B)
    seq = state.get("tool_call_sequence", [])
    if len(seq) >= 4:
        a, b, c, d = seq[-4:]
        if a != b and a == c and b == d:
            print(f"\n🚨 [熔断] 检测到震荡死循环: {a} -> {b} -> {c} -> {d}，强制结束。")
            return "validate"

    return "tools"


def trim_messages_node(state: State) -> dict:
    """裁剪消息防止 token 爆炸 + 新对话时重置工具状态"""
    messages = state.get("messages", [])

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_sys_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    MAX_HISTORY = 10
    trimmed_non_sys = non_sys_msgs[-MAX_HISTORY:]
    trimmed_messages = system_msgs + trimmed_non_sys

    if trimmed_messages and isinstance(trimmed_messages[-1], HumanMessage):
        print("\n🔄 [Reset] 检测到新用户提问，重置工具调用状态")
        return {
            "messages": trimmed_messages,
            "tool_call_sequence": [],
            "tool_call_count": 0,
        }

    return {"messages": trimmed_messages}


# ========== 构建图 ==========
builder = StateGraph(State)

builder.add_node("trim_messages", trim_messages_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("validate", validate)

builder.add_edge(START, "trim_messages")
builder.add_edge("trim_messages", "agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "validate": "validate",
    },
)
# 工具执行后直接回到 agent，让其查看结果并决定下一步
builder.add_edge("tools", "agent")
# 验证后结束
builder.add_edge("validate", END)

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
        print("\n👋 手动中断，退出对话，再见！")
    except Exception as e:
        print(f"\n❌ 运行异常: {e}")
        raise


if __name__ == "__main__":
    run()
