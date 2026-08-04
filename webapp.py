"""
RAG 系统 Web 前端 — FastAPI 后端
=================================
启动: python webapp.py
访问: http://localhost:8000
"""

import os
import sys
import json
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── 项目路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── 全局状态 ──
app_state = {
    "retriever": None,
    "tools": None,
    "mcp_tools": [],
    "pipeline": None,
    "archive_store": None,
    "metadata_store": None,
    "ready": False,
    "status_msg": "正在初始化...",
}

# ── 生命周期 ──
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型和装配系统"""
    try:
        app_state["status_msg"] = "正在加载 Embedding 模型..."
        print("[启动] 加载 Embedding 模型...")

        from build import build
        retriever, tools = build()

        app_state["retriever"] = retriever
        app_state["tools"] = tools
        app_state["ready"] = True
        app_state["status_msg"] = "系统就绪"

        # 从工具对象中提取存储引用（每个工具都持有对应组件的引用）
        for tool in tools:
            name = tool.name
            if name == "rag_ingest_file" and hasattr(tool, 'pipeline'):
                app_state["pipeline"] = tool.pipeline
            elif name == "rag_list_archive" and hasattr(tool, 'archive_store'):
                app_state["archive_store"] = tool.archive_store
            elif name == "rag_list_active_files" and hasattr(tool, 'metadata_store'):
                app_state["metadata_store"] = tool.metadata_store

        # 异步加载 MCP 远程工具（lifespan 本身是 async，直接 await）
        try:
            from mcp_client import load_remote_tools
            app_state["mcp_tools"] = await load_remote_tools()
            if app_state["mcp_tools"]:
                print(f"[启动] MCP 远程工具: {[t.name for t in app_state['mcp_tools']]}")
        except Exception as e:
            print(f"[启动] MCP 加载失败（不影响本地功能）: {e}")
            app_state["mcp_tools"] = []

        print(f"[启动] 系统就绪，加载了 {len(tools)} 个本地工具 + {len(app_state['mcp_tools'])} 个远程工具")

    except Exception as e:
        app_state["status_msg"] = f"启动失败: {str(e)}"
        print(f"[启动] 错误: {e}")
        raise

    yield  # 应用运行期间

    # 关闭时清理
    print("[关闭] 系统关闭")


# ── FastAPI 应用 ──
app = FastAPI(title="RAG 知识库系统", version="1.0", lifespan=lifespan)

TEMPLATES_DIR = PROJECT_ROOT / "templates"


# ======================================================================
# 设置 API
# ======================================================================
_SETTINGS_FILE = PROJECT_ROOT / ".rag_settings.json"

DEFAULT_SETTINGS = {
    "api_base": os.getenv("URL", "https://api.deepseek.com"),
    "api_key": os.getenv("API", ""),
    "embed_model": "deepseek-embedding-base",
    "embed_mode": "api",
    "llm_model": os.getenv("MODEL", "deepseek-v4-flash"),
}


def _read_settings():
    if _SETTINGS_FILE.exists():
        try:
            return json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_settings(data: dict):
    current = _read_settings()
    current.update(data)
    _SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/settings")
async def get_settings():
    """读取当前设置"""
    stored = _read_settings()
    merged = {**DEFAULT_SETTINGS, **stored}
    # 隐藏 API key 的部分内容
    if merged.get("api_key"):
        key = merged["api_key"]
        if len(key) > 8:
            merged["api_key_masked"] = key[:4] + "****" + key[-4:]
        else:
            merged["api_key_masked"] = "****"
    return {"success": True, "settings": merged}


@app.post("/api/settings")
async def save_settings(request: Request):
    """保存设置"""
    body = await request.json()
    allowed = {"api_base", "api_key", "embed_model", "embed_mode", "llm_model",
               "local_model_path", "rerank_model_path"}
    data = {k: v for k, v in body.items() if k in allowed and v is not None}
    _write_settings(data)
    # 同时更新 os.environ 以立即生效
    if "api_key" in data:
        os.environ["API"] = data["api_key"]
    if "api_base" in data:
        os.environ["URL"] = data["api_base"]
    return {"success": True, "message": "设置已保存，重启后生效。（API key 已立即生效）"}


# ======================================================================
# 页面
# ======================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>index.html 未找到，请检查 templates/ 目录</h1>", status_code=404)


@app.get("/api/status")
async def status():
    """系统状态"""
    return {
        "ready": app_state["ready"],
        "status": app_state["status_msg"],
        "tools": [t.name for t in (app_state["tools"] or [])],
    }


# ======================================================================
# 智能问答 (Agent 模式 — SSE 流式)
# ======================================================================
@app.post("/api/chat")
async def chat(request: Request):
    """AI 问答 — Agent 自主调用工具，SSE 流式返回"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "消息不能为空"}, status_code=400)

    async def generate():
        from langchain.agents import create_agent

        # 合并本地工具 + MCP 远程工具
        all_tools = list(app_state["tools"] or []) + list(app_state["mcp_tools"] or [])
        if not all_tools:
            yield f"data: {json.dumps({'type': 'error', 'text': '没有可用工具'})}\n\n"
            return

        llm = ChatOpenAI(
            model=os.getenv("MODEL", "deepseek-v4-flash"),
            openai_api_key=os.getenv("API"),
            openai_api_base=os.getenv("URL", "https://api.deepseek.com"),
            temperature=0.3,
            streaming=True,
        )

        system_prompt = """你是企业内部智能助手，可以自主调用工具来完成任务。
- 查内部文档/制度 -> 知识库检索工具
- 上传/更新/删除文件 -> 知识库管理工具
- GitHub 操作 -> GitHub 工具
- 通用知识 -> 直接回答"""

        agent = create_agent(model=llm, tools=all_tools, system_prompt=system_prompt)

        try:
            # 先获取 Agent 执行过程的中间步骤
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_message)]}
            )
            # 提取最终回答
            messages = result.get("messages", [])
            for msg in messages:
                # 显示工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get('name', 'unknown')
                        args = str(tc.get('args', {}))[:300]
                        yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'input': args})}\n\n"
                # 显示工具结果
                if hasattr(msg, 'name') and hasattr(msg, 'content') and not hasattr(msg, 'tool_calls'):
                    summary = str(msg.content)[:300]
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': msg.name, 'output': summary})}\n\n"

            # 最终回答（最后一条 AIMessage）
            final_msg = messages[-1]
            answer = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
            yield f"data: {json.dumps({'type': 'token', 'text': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'full': answer})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'text': f'执行出错: {str(e)[:300]}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ======================================================================
# 文档入库
# ======================================================================
@app.post("/api/ingest")
async def ingest(
    file: UploadFile = File(...),
    doc_code: str = Form(...),
    uploader_id: str = Form("system"),
):
    """单文件入库"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    try:
        # 读取上传文件内容
        content = await file.read()

        # 保存到临时目录再交给 pipeline
        tmp_path = PROJECT_ROOT / "storage" / "tmp_uploads"
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_path / file.filename
        tmp_file.write_bytes(content)

        # 调用 ingest 工具
        ingest_tool = next((t for t in app_state["tools"] if t.name == "rag_ingest_file"), None)
        if ingest_tool is None:
            return JSONResponse({"error": "找不到入库工具"}, status_code=500)

        result = ingest_tool._run(
            filename=str(tmp_file),
            doc_code=doc_code,
            uploader_id=uploader_id,
        )

        # 清理临时文件
        try:
            tmp_file.unlink()
        except Exception:
            pass

        success = result.startswith("✅")
        return {
            "success": success,
            "message": result,
        }

    except Exception as e:
        return JSONResponse({"success": False, "message": f"上传失败: {str(e)}"}, status_code=500)


@app.post("/api/ingest-folder")
async def ingest_folder(request: Request):
    """批量文件夹导入"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    body = await request.json()
    folder_path = body.get("folder_path", "").strip()
    recursive = body.get("recursive", True)

    if not folder_path or not Path(folder_path).exists():
        return JSONResponse({"success": False, "message": "文件夹路径不存在"}, status_code=400)

    try:
        folder_tool = next((t for t in app_state["tools"] if t.name == "rag_ingest_folder"), None)
        if folder_tool is None:
            return JSONResponse({"error": "找不到批量导入工具"}, status_code=500)

        result = folder_tool._run(
            folder_path=folder_path,
            uploader_id="web_user",
            recursive=recursive,
        )
        return {"success": True, "message": result}
    except Exception as e:
        return JSONResponse({"success": False, "message": f"批量导入失败: {str(e)}"}, status_code=500)


# ======================================================================
# 知识检索
# ======================================================================
@app.post("/api/retrieve")
async def retrieve(request: Request):
    """检索知识库"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    body = await request.json()
    query = body.get("query", "").strip()
    top_n = body.get("top_n", 5)

    if not query:
        return JSONResponse({"error": "查询不能为空"}, status_code=400)

    try:
        retriever = app_state["retriever"]
        docs = retriever.invoke(query)

        results = []
        for i, d in enumerate(docs[:top_n]):
            results.append({
                "index": i + 1,
                "source": d.metadata.get("source", "未知"),
                "chunk_id": d.metadata.get("chunk_id", ""),
                "score": round(d.metadata.get("score", 0), 4),
                "content": d.page_content[:300],
            })

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


# ======================================================================
# 文件管理
# ======================================================================
@app.get("/api/files")
async def list_files(keyword: str = Query("")):
    """活跃文件列表"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    try:
        list_tool = next((t for t in app_state["tools"] if t.name == "rag_list_active_files"), None)
        if list_tool is None:
            return JSONResponse({"error": "找不到文件列表工具"}, status_code=500)

        result = list_tool._run(keyword=keyword)

        # 解析工具返回的文本为结构化数据
        files = []
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("📄"):
                # 格式: 📄 filename | doc_code=CODE | id=FILE_ID
                parts = line.replace("📄 ", "").split(" | ")
                entry = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        entry[k.strip()] = v.strip()
                    else:
                        entry["original_filename"] = p.strip()
                files.append(entry)

        return {"success": True, "files": files, "raw": result}
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.get("/api/archive/{file_id}")
async def archive_history(file_id: str):
    """归档历史"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    try:
        archive_tool = next((t for t in app_state["tools"] if t.name == "rag_list_archive"), None)
        if archive_tool is None:
            return JSONResponse({"error": "找不到归档工具"}, status_code=500)

        result = archive_tool._run(file_id=file_id)

        versions = []
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("📦"):
                versions.append(line.replace("📦", "").strip())

        return {"success": True, "file_id": file_id, "versions": versions, "raw": result}
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.post("/api/update")
async def update_file(
    file: UploadFile = File(...),
    doc_code: str = Form(...),
):
    """更新文件（归档旧版 + 入库新版）"""
    if not app_state["ready"]:
        return JSONResponse({"error": "系统尚未就绪"}, status_code=503)

    try:
        content = await file.read()

        tmp_path = PROJECT_ROOT / "storage" / "tmp_uploads"
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_path / file.filename
        tmp_file.write_bytes(content)

        update_tool = next((t for t in app_state["tools"] if t.name == "rag_update_file"), None)
        if update_tool is None:
            return JSONResponse({"error": "找不到更新工具"}, status_code=500)

        result = update_tool._run(
            doc_code=doc_code,
            new_file_path=str(tmp_file),
        )

        try:
            tmp_file.unlink()
        except Exception:
            pass

        success = result.startswith("✅")
        return {"success": success, "message": result}

    except Exception as e:
        return JSONResponse({"success": False, "message": f"更新失败: {str(e)}"}, status_code=500)


# ======================================================================
# 启动入口
# ======================================================================
if __name__ == "__main__":
    import webbrowser
    import threading

    print("=" * 60)
    print("  RAG 知识库系统 — Web 前端")
    print("  http://localhost:8000")
    print("=" * 60)

    # 自动打开浏览器
    def open_browser():
        import time
        time.sleep(2)  # 等服务器启动
        webbrowser.open("http://localhost:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    # 直接传 app 对象（兼容 PyInstaller 打包）
    import socket
    PORT = 8000
    # 如果 8000 被占用，自动换端口
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sock.connect_ex(('127.0.0.1', PORT)) == 0:
        for p in range(8001, 8010):
            if sock.connect_ex(('127.0.0.1', p)) != 0:
                PORT = p
                break
    sock.close()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
    )
