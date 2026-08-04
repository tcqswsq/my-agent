"""
异步入库任务管理器
==================
解决大文件上传时阻塞事件循环的问题：
- POST /api/ingest 立即返回 task_id
- GET /api/ingest/{task_id}/status 轮询进度
- 线程池后台执行实际的入库操作
"""

import time
import uuid
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class IngestTaskManager:
    """异步入库：提交 → 后台执行 → 轮询结果"""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def submit(self, pipeline, file_bytes: bytes, filename: str,
               doc_code: str, uploader_id: str = "system",
               category: str = "其他", tags: str = "") -> str:
        """提交入库任务，立即返回 task_id"""
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        task = {
            "task_id": task_id,
            "filename": filename,
            "doc_code": doc_code,
            "status": "pending",
            "progress": 0,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._tasks[task_id] = task

        future = self._executor.submit(
            self._run_sync, task, pipeline, file_bytes, filename,
            doc_code, uploader_id, category, tags
        )
        task["_future"] = future
        return task_id

    def get_status(self, task_id: str) -> Optional[dict]:
        """查询任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return None
        # 检查线程是否已完成
        future = task.get("_future")
        if future and future.done() and task["status"] == "processing":
            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            try:
                result = future.result()
                task["status"] = "done"
                task["result"] = result
                task["progress"] = 100
                task["updated_at"] = now
            except Exception as e:
                task["status"] = "error"
                task["error"] = str(e)
                task["updated_at"] = now
        return {k: v for k, v in task.items() if not k.startswith("_")}

    @staticmethod
    def _run_sync(task: dict, pipeline, file_bytes: bytes, filename: str,
                  doc_code: str, uploader_id: str, category: str, tags: str) -> dict:
        """在线程池中执行同步入库（静态方法，避免序列化 self）"""
        task["status"] = "processing"
        task["progress"] = 10
        task["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        result = pipeline.ingest_file(
            file_bytes, filename, uploader_id,
            doc_code=doc_code, category=category, tags=tags
        )
        return {"file_id": result["file_id"], "chunks_count": result["chunks_count"]}

    # ── 单例 ──
    _instance: Optional["IngestTaskManager"] = None

    @classmethod
    def get_instance(cls) -> "IngestTaskManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_task_manager() -> IngestTaskManager:
    return IngestTaskManager.get_instance()
