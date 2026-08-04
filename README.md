# RAG System

企业级本地知识库系统，基于 RAG (Retrieval-Augmented Generation) 架构，支持多格式文档入库、混合检索、版本管理与一键回滚。提供 Web 界面和 LangChain Agent 两种使用方式。

## 快速开始

### 1. 环境要求

- Python 3.9 - 3.11
- 推荐 16G+ 内存
- 可选 GPU（加速向量化与重排序）

### 2. 安装依赖

```bash
pip install -r requirements-api.txt
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
API=你的DeepSeek_API_Key
URL=https://api.deepseek.com
MODEL=deepseek-v4-flash
```

或者通过 Web 界面「系统设置」面板填写，设置会自动保存到 `.rag_settings.json`。

### 4. 启动

**Web 界面（推荐）：**

```bash
python webapp.py
```

访问 `http://localhost:8000`，浏览器会自动打开。

**命令行 Agent：**

```bash
python mian.py
```

正常启动后显示加载了 **6 个工具**。

## 项目结构

```
rag_system/
├── core/                     # 核心引擎
│   ├── stores.py             # 存储层（文件/元数据/向量/归档）
│   ├── pipeline.py           # 离线处理管道（解析→清洗→分块→入库）
│   ├── embedder.py           # 统一嵌入层（API / 本地模式）
│   ├── reranker.py           # 重排序器
│   └── retriever.py          # 混合检索（向量 + BM25 + 重排序）
├── tools/                    # LangChain 工具封装
│   ├── ingest.py             # 单文件 / 批量上传
│   ├── retrieve.py           # 知识库检索
│   └── manage.py             # 更新 / 归档 / 列出活跃文件
├── utils/
│   └── processors.py         # 文档清洗与分块算法
├── templates/
│   └── index.html            # Web 前端页面
├── config.py                 # 全局配置
├── build.py                  # 一键装配
├── mian.py                   # 命令行 Agent 入口
├── webapp.py                 # Web 界面入口
├── requirements-api.txt      # Python 依赖
├── .gitignore
└── README.md
```

## 可用工具（6 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `rag_ingest_file` | 上传单个文件 | `filename`, `doc_code`, `uploader_id` |
| `rag_ingest_folder` | 批量上传文件夹 | `folder_path`, `recursive` |
| `rag_retrieve` | 检索知识库 | `query`, `top_n` |
| `rag_update_file` | 更新文件（自动归档旧版） | `doc_code`, `new_file_path` |
| `rag_list_archive` | 查看归档历史 | `file_id` |
| `rag_list_active_files` | 列出活跃文件 | `keyword`（可选，按文件名过滤） |

## 接入 LangChain Agent

```python
from build import build
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

retriever, tools = build()
llm = ChatOpenAI(model="deepseek-v4-flash")

agent = create_tool_calling_agent(llm, tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "年假怎么请？"})
```

## 版本管理说明

- **更新文件**：通过 `doc_code`（业务唯一编码）定位文件，自动将旧版归档至 `storage/archive/<file_id>/`，默认保留 30 天（可在 `config.py` 中调整 `ARCHIVE_TTL_DAYS`）
- **停用旧版**：数据库中标记 `is_active=0`，不参与检索
- **定时清理**：调用 `archive.cleanup_expired()` 删除过期归档

## 支持的文件格式

PDF, DOCX, TXT, MD, XLSX, CSV, PPTX, HTML, JSON

## 嵌入模式

支持两种嵌入模式，通过 `.rag_settings.json` 或 `config.py` 切换：

| 模式 | 说明 |
|------|------|
| `api`（默认） | 调用 DeepSeek Embedding API，无需本地模型，开箱即用 |
| `local` | 使用本地 SentenceTransformer 模型，完全离线运行 |

## 详细文档

请参阅 `USER_MANUAL.md` 获取完整使用说明。

## License

Internal Use Only
