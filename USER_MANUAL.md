# 📘 RAG 企业级知识库系统 - 部署与使用说明书

## 📌 1. 系统简介
本系统是一套基于 **RAG (Retrieval-Augmented Generation)** 架构的企业级本地知识库解决方案。
- **本地私有化**：所有数据（文档、向量、元数据）均存储在本地，无数据泄露风险。
- **混合检索**：结合向量语义检索（Dense）与关键词检索（BM25），并经过 Cross-Encoder 重排序，准确率极高。
- **全生命周期管理**：支持文件上传、批量导入、版本更新、历史归档及一键回滚。
- **多格式支持**：PDF, Word, Excel, PPT, TXT, MD, HTML, JSON 等。

---

## 📂 2. 项目结构说明
请严格按照以下结构存放文件，否则会导致模块导入错误。

```text
rag_system/
├── core/                     # 核心引擎（不建议修改）
│   ├── __init__.py
│   ├── stores.py             # 存储层：文件、元数据(SQLite)、向量(Chroma)、归档
│   ├── pipeline.py           # 离线处理：解析、切片、入库
│   └── retriever.py          # 在线检索：混合检索与重排序逻辑
├── tools/                    # LangChain 工具集（AI Agent 调用入口）
│   ├── __init__.py
│   ├── ingest.py             # 工具：单文件/文件夹上传
│   ├── retrieve.py           # 工具：知识库检索
│   └── manage.py             # 工具：更新、归档列表
├── utils/                    # 工具函数
│   ├── __init__.py
│   └── processors.py         # 文档清洗、分块、语义合并算法
├── docs/                     # 文档目录
│   └── USER_MANUAL.md        # 本文件
├── storage/                  # 【自动生成】数据存储目录（.gitignore 已忽略）
│   ├── originals/            # 原始文件
│   ├── vectors/              # Chroma 向量数据库
│   ├── archive/              # 旧版本文件归档
│   └── metadata.db           # SQLite 元数据
├── config.py                 # 全局配置文件（路径、模型地址）
├── build.py                  # 系统装配脚本
├── main.py                   # 程序入口/测试脚本
├── requirements.txt          # Python 依赖列表
├── .gitignore
└── README.md
```

---

## 🚀 3. 环境准备与安装

### 3.1 环境要求
- Python 3.9 - 3.11
- 足够的内存（推荐 16G+）
- （可选）GPU（大幅提升向量化与重排序速度）

### 3.2 安装依赖
在项目根目录 (`rag_system/`) 下执行：

```bash
pip install -r requirements.txt
```

### 3.3 模型准备
请确保你已经下载了以下模型到本地（或修改为在线模型名称）：
1. **Embedding 模型**：`paraphrase-multilingual-MiniLM-L12-v2`
2. **Reranker 模型**：`BAAI/bge-reranker-base`

并在 `config.py` 中核对路径是否正确。

---

## ⚙️ 4. 配置修改
打开 `config.py`，根据你的环境修改以下变量：

```python
# config.py

# 模型路径（必须修改为你的本地路径）
EMBED_MODEL_PATH = r"E:\ana\MiniLM-L12-v2\sentence-transformers\paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL_PATH = r"E:\ana\hf1\BAAI\bge-reranker-base"

# 存储路径（一般无需修改）
STORAGE_DIR = "storage"

# 归档策略（天）
ARCHIVE_TTL_DAYS = 30
```

---

## 🛠️ 5. 核心功能使用指南

### 5.1 启动系统
运行 `main.py` 验证系统是否正常加载：
```bash
python main.py
```
正常输出应显示加载了 5 个工具。

### 5.2 数据入库（Ingestion）

**单文件上传**
AI Agent 会调用 `rag_ingest_file` 工具：
- **参数**：`filename` (文件路径), `uploader_id` (上传人)
- **示例**：上传 `员工手册.pdf`

**批量文件夹上传**
AI Agent 会调用 `rag_ingest_folder` 工具：
- **参数**：`folder_path` (文件夹路径), `recursive` (是否递归子目录)
- **示例**：导入 `E:/company_docs/hr/` 目录下所有文件。

### 5.3 知识检索（Retrieval）
AI Agent 会调用 `rag_retrieve` 工具：
- **参数**：`query` (用户问题), `top_n` (返回几条结果)
- **示例**：查询"年假怎么请？"
- **返回**：包含来源文件名、页码和相关内容的摘要。

### 5.4 文件更新与版本管理（核心亮点）

**更新文件（自动归档旧版）**
当需要把"2025版制度"替换为"2026版"时，AI 调用 `rag_update_file`：
- **参数**：`old_file_id` (旧文件ID), `new_file_path` (新文件路径)
- **背后逻辑**：
    1. 将旧文件复制到 `storage/archive/` 目录。
    2. 数据库标记旧文件为 `is_active=0`。
    3. 删除旧文件的向量索引。
    4. 解析并入库新文件。

**查看归档历史**
- **工具**：`rag_list_archive`
- **参数**：`file_id`
- **用途**：审计追溯，查看文件变更记录。

**定时清理**
在 `main.py` 或定时任务中调用：
```python
from core.stores import ArchiveStore
from config import ARCHIVE_STORE_PATH, ARCHIVE_TTL_DAYS
archive = ArchiveStore(ARCHIVE_STORE_PATH, ARCHIVE_TTL_DAYS)
deleted_count = archive.cleanup_expired()
print(f"清理了 {deleted_count} 个过期归档。")
```

---

## 🤖 6. 接入 AI Agent (LangChain / OpenAI Function Calling)

系统通过 `build.py` 返回一个标准的 LangChain Tool 列表。

```python
from build import build

retriever, tools = build()

# 将这些 tools 绑定给你的 LLM
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o")
agent = create_tool_calling_agent(llm, tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "请帮我查找关于信息安全的最新规定"})
```

---

## 🐛 7. 常见问题排查 (FAQ)

**Q1: 运行时报错 `ModuleNotFoundError: No module named 'utils'`**
- **原因**：Python 路径问题。
- **解决**：
    1. 确保在 `rag_system` 目录下运行 `python main.py`。
    2. 检查 `core/pipeline.py` 中是否有 `from ..utils.processors import ...`。
    3. 在 IDE（如 PyCharm）中将 `rag_system` 文件夹标记为 `Sources Root`。

**Q2: PDF 解析乱码或无文字**
- **原因**：PDF 是扫描件（图片），缺少 OCR 配置。
- **解决**：当前系统主要针对可复制文本的 PDF。如需处理扫描件，需在 `pipeline.py` 的 `UnstructuredPDFLoader` 中配置 `ocr_languages="chi_sim"` 并安装相应 OCR 引擎。

**Q3: 内存占用过高**
- **原因**：Embedding 模型和 Reranker 模型较大。
- **解决**：
    1. 确保 `config.py` 中模型路径正确。
    2. 如果是 CPU 运行，这是正常现象。
    3. 考虑使用更小的模型（如 `all-MiniLM-L6-v2`）。

**Q4: 更新文件后，检索还能搜到旧内容？**
- **原因**：Chroma 向量删除可能有延迟，或 BM25 缓存未更新。
- **解决**：系统已自动处理。如遇极端情况，删除 `storage/bm25_cache.json` 并重启，系统会自动重建缓存。

---

## 📈 8. 维护与监控
1. **监控 `storage/` 目录大小**：特别是 `vectors` 和 `archive` 文件夹。
2. **定期备份**：务必备份 `metadata.db` 和 `originals` 文件夹。向量库 `vectors` 可通过重新运行 ingestion 重建，但原始文件丢了就无法恢复了。
3. **日志**：建议在 `stores.py` 和 `pipeline.py` 中加入 `logging` 模块，记录入库和检索的详细信息，便于排查问题。

---
📧 **技术支持**：请联系内部开发团队。
📅 **最后更新**：2026-01-15
