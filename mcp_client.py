"""
MCP 客户端 — 连接外部 MCP 服务器，加载别人的 skill
=================================================
启动: python mcp_client.py
配置: 编辑 mcp_config.json 添加要连接的 MCP 服务器

用法:
    from mcp_client import load_all_tools
    tools = await load_all_tools()       # 本地 RAG 工具 + 远程 MCP 工具
    tools = await load_all_tools(local_tools_only=True)  # 仅本地工具
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── 配置文件路径 ──
MCP_CONFIG_PATH = PROJECT_ROOT / "mcp_config.json"


def _load_mcp_config() -> dict:
    """读取 MCP 服务器配置，过滤掉 _ 开头的注释条目"""
    if not MCP_CONFIG_PATH.exists():
        print(f"[MCP] 配置文件 {MCP_CONFIG_PATH} 不存在，使用空配置")
        return {}
    try:
        config = json.loads(MCP_CONFIG_PATH.read_text(encoding="utf-8"))
        servers = config.get("mcpServers", {})
        # 只保留有效的服务器配置（必须有 transport 字段），过滤掉纯注释条目
        return {k: v for k, v in servers.items() if isinstance(v, dict) and "transport" in v}
    except Exception as e:
        print(f"[MCP] 读取配置失败: {e}")
        return {}


async def load_remote_tools(server_configs: Optional[dict] = None) -> list:
    """
    连接远程 MCP 服务器，加载它们的工具/skill

    Args:
        server_configs: 服务器配置字典，格式:
            {
                "server_name": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["path/to/server.py"],
                }
            }
            不传则从 mcp_config.json 读取

    Returns:
        远程 MCP 工具列表
    """
    if server_configs is None:
        server_configs = _load_mcp_config()

    if not server_configs:
        print("[MCP] 没有配置任何远程 MCP 服务器")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        print(f"[MCP] 正在连接 {len(server_configs)} 个远程服务器: {list(server_configs.keys())}")
        client = MultiServerMCPClient(server_configs)
        tools = await client.get_tools()
        print(f"[MCP] 已加载 {len(tools)} 个远程工具: {[t.name for t in tools]}")
        return tools

    except ImportError:
        print("[MCP] 缺少 langchain-mcp-adapters，请安装: pip install langchain-mcp-adapters")
        return []
    except Exception as e:
        # 逐个尝试，单个失败不影响其他
        print(f"[MCP] 批量连接失败: {e}，逐个重试...")
        all_tools = []
        for name, cfg in server_configs.items():
            try:
                client = MultiServerMCPClient({name: cfg})
                tools = await client.get_tools()
                all_tools.extend(tools)
                print(f"[MCP]   {name}: {len(tools)} 个工具")
            except Exception as e2:
                print(f"[MCP]   {name}: 连接失败 ({e2})")
        return all_tools


async def load_all_tools(local_tools_only: bool = False) -> list:
    """
    加载所有工具：本地 RAG 工具 + 远程 MCP 工具

    Args:
        local_tools_only: 仅加载本地工具，不连接远程服务器

    Returns:
        合并后的工具列表
    """
    # 1. 本地 RAG 工具
    # 兼容直接运行 (python mcp_client.py) 和包内导入两种情况
    try:
        from build import build
    except ImportError:
        from .build import build
    print("[MCP] 加载本地 RAG 系统...")
    _, local_tools = build()
    print(f"[MCP] 本地工具: {[t.name for t in local_tools]}")

    if local_tools_only:
        return local_tools

    # 2. 远程 MCP 工具
    remote_tools = await load_remote_tools()

    # 3. 合并
    all_tools = list(local_tools) + list(remote_tools)
    print(f"[MCP] 工具总数: {len(all_tools)}（本地 {len(local_tools)} + 远程 {len(remote_tools)}）")

    return all_tools


# ── 命令行测试入口 ──
async def main():
    """测试：加载所有工具并列出"""
    print("=" * 60)
    print("  MCP 客户端 — 工具加载测试")
    print("=" * 60)

    tools = await load_all_tools()

    print("\n📋 可用工具一览：")
    for i, t in enumerate(tools, 1):
        desc = (t.description or "")[:80]
        print(f"  {i}. {t.name}")
        print(f"     {desc}")

    print("\n✅ MCP 客户端就绪")


if __name__ == "__main__":
    asyncio.run(main())
