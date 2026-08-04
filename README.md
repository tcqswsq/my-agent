# RAG System

企业级本地知识库系统，基于 RAG (Retrieval-Augmented Generation) 架构，支持多格式文档入库、混合检索、版本管理与一键回滚。

## 快速开始

### 1. 环境要求

- Python 3.9 - 3.11
- 推荐 16G+ 内存
- 可选 GPU（加速向量化与重排序）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型路径

编辑 `config.py`，修改 Embedding 和 Reranker 模型的本地路径：

```python
EMBED_MODEL_PATH = "path/to/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_PATH = "path/to/bge-reranker-base"
```

### 4. 启动

```bash
python main.py
```

正常输出应显示加载了 5 个工具。

## 项目结构

```
rag_system/
├── core/                     # 核心引擎
│   ├── stores.py             # 存储层（文件/元数据/向量/归档）
│   ├── pipeline.py           # 离线处理管道（解析→清洗→分块→入库）
│   └── retriever.py          # 混合检索（向量+BM25+重排序）
├── tools/                    # LangChain 工具封装
│   ├── ingest.py             # 单文件/批量上传
│   ├── retrieve.py           # 知识库检索
│   └── manage.py             # 更新/归档/回滚
├── utils/
│   └── processors.py         # 文档清洗与分块算法
├── config.py                 # 全局配置
├── build.py                  # 一键装配
├── main.py                   # 入口
└── requirements.txt
```

## 可用工具（AI Agent 调用）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `rag_ingest_file` | 上传单个文件 | `filename`, `uploader_id` |
| `rag_ingest_folder` | 批量上传文件夹 | `folder_path`, `recursive` |
| `rag_retrieve` | 检索知识库 | `query`, `top_n` |
| `rag_update_file` | 更新文件（自动归档旧版） | `old_file_id`, `new_file_path` |
| `rag_list_archive` | 查看归档历史 | `file_id` |

## 接入 LangChain Agent

```python
from build import build
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

retriever, tools = build()
llm = ChatOpenAI(model="gpt-4o")

agent = create_tool_calling_agent(llm, tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "年假怎么请？"})
```

## 版本管理说明

- **更新文件**：自动将旧文件归档至 `storage/archive/<file_id>/`，保留 30 天（可在 `config.py` 调整）
- **停用旧版**：数据库中标记 `is_active=0`，不参与检索
- **定时清理**：调用 `archive.cleanup_expired()` 删除过期归档

## 支持的文件格式

PDF, DOCX, TXT, MD, XLSX, CSV, PPTX, HTML, JSON

## 详细文档

请参阅 `docs/USER_MANUAL.md` 获取完整使用说明。

## License

Internal Use Only
