"""
统一嵌入层 — 支持 API 模式和本地模式
======================================
API 模式: 调用 DeepSeek /v1/embeddings（默认）
本地模式: 使用 SentenceTransformer 本地推理
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Union
import numpy as np


def _load_settings():
    """读取 .rag_settings.json"""
    settings_path = Path(__file__).resolve().parent.parent / ".rag_settings.json"
    if settings_path.exists():
        try:
            return json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_embed_config():
    """获取当前嵌入配置"""
    settings = _load_settings()
    return {
        "mode": settings.get("embed_mode", "api"),  # "api" | "local"
        "api_base": settings.get("api_base", os.getenv("URL", "https://api.deepseek.com")),
        "api_key": settings.get("api_key", os.getenv("API", "")),
        "embed_model": settings.get("embed_model", "deepseek-embedding-base"),
        "local_model_path": settings.get("local_model_path", ""),
    }


class Embedder:
    """统一嵌入接口"""

    def __init__(self, mode: str = None):
        """
        mode: "api" | "local" | None(自动检测)
        """
        config = get_embed_config()
        if mode is None:
            mode = config["mode"]

        # 自动检测：如果有本地模型路径就优先用本地
        if mode == "local" or config["local_model_path"]:
            self._init_local(config)
        else:
            self._init_api(config)

    def _init_api(self, config: dict):
        """初始化 API 模式"""
        self.mode = "api"
        self.api_base = config["api_base"].rstrip("/")
        self.api_key = config["api_key"]
        self.embed_model = config.get("embed_model", "deepseek-embedding-base")
        self.dim = 1536  # DeepSeek embedding 维度

        import requests
        self._requests = requests

    def _init_local(self, config: dict):
        """初始化本地模式"""
        self.mode = "local"
        model_path = config.get("local_model_path", "")
        if not model_path:
            # 尝试从 config.py 读取
            try:
                from config import EMBED_MODEL_PATH
                model_path = EMBED_MODEL_PATH
            except ImportError:
                pass

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"本地 Embedding 模型未找到: {model_path}\n"
                f"请在设置中配置正确的路径，或切换到 API 模式。"
            )

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_path)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Union[str, List[str]], convert_to_tensor=False,
               show_progress_bar=False, **kwargs) -> np.ndarray:
        """
        与 SentenceTransformer.encode() 接口兼容
        返回 numpy array: (n_texts, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.mode == "api":
            return self._encode_api(texts)
        else:
            return self._encode_local(texts, convert_to_tensor, show_progress_bar)

    def _encode_api(self, texts: List[str]) -> np.ndarray:
        """通过 DeepSeek API 获取嵌入向量"""
        embeddings = []
        # 分批处理，每批最多 8 条
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = self._requests.post(
                    f"{self.api_base}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.embed_model,
                        "input": batch,
                    },
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Embedding API 错误 ({resp.status_code}): {resp.text[:200]}")

                data = resp.json()
                for item in data["data"]:
                    embeddings.append(np.array(item["embedding"], dtype=np.float32))

            except Exception as e:
                raise RuntimeError(f"Embedding API 调用失败: {e}")

        return np.stack(embeddings)

    def _encode_local(self, texts: List[str], convert_to_tensor, show_progress_bar) -> np.ndarray:
        """本地模型推理"""
        return self._model.encode(
            texts,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=show_progress_bar,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim
