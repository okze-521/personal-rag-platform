"""
Embedding 客户端 — 通过 Ollama API 调用 BGE-M3 生成文本向量
"""
from __future__ import annotations

import logging
import time

import requests

from src.config import settings

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    BGE-M3 向量化客户端 (基于 Ollama REST API)

    使用方式:
        embedder = OllamaEmbedder()
        vec = embedder.embed("这是一段测试文本")
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url or f"http://127.0.0.1:11434"
        self.model = model or settings.EMBEDDING_MODEL
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/embeddings"

    def embed(self, text: str) -> list[float]:
        """单条文本向量化"""
        resp = requests.post(
            self._endpoint,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str], delay: float = 0.1) -> list[list[float]]:
        """
        批量向量化 (逐条调用，带间隔防限流)

        Args:
            texts: 文本列表
            delay: 请求间隔 (秒)

        Returns:
            向量列表，与输入一一对应
        """
        vectors: list[list[float]] = []
        total = len(texts)

        for i, text in enumerate(texts, 1):
            try:
                vec = self.embed(text)
                vectors.append(vec)
                logger.debug(f"Embedded {i}/{total} | dim={len(vec)}")
            except Exception as e:
                logger.error(f"Embed failed at {i}/{total}: {e}")
                vectors.append([])  # 占位

            if i < total:
                time.sleep(delay)

        success = sum(1 for v in vectors if v)
        logger.info(f"Batch embed done: {success}/{total} successful")
        return vectors
