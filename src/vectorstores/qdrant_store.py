"""
Qdrant 检索器 — 向量相似度搜索封装
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """单条检索命中结果"""
    id: str
    score: float
    text: str
    source: str
    chunk_index: int
    payload: dict[str, Any]


class QdrantStore:
    """
    Qdrant 检索器

    使用方式:
        store = QdrantStore()
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.qdrant_url
        self.collection_name = collection_name or settings.VECTOR_STORE_NAME

    @property
    def _search_url(self) -> str:
        return f"{self.base_url}/collections/{self.collection_name}/points/search"

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchHit]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量 (1024维)
            top_k: 返回结果数
            score_threshold: 最低相似度阈值 (0~1)

        Returns:
            命中结果列表，按相似度降序排列
        """
        if not query_vector:
            logger.warning("Empty query vector, returning empty result")
            return []

        resp = requests.post(
            self._search_url,
            json={
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
                "score_threshold": score_threshold,
            },
            timeout=10,
        )
        resp.raise_for_status()

        results = resp.json().get("result", [])
        hits = []

        for item in results:
            payload = item.get("payload", {})
            hits.append(SearchHit(
                id=item.get("id", ""),
                score=item.get("score", 0.0),
                text=payload.get("text", payload.get("content", "")),
                source=payload.get("source", payload.get("doc_id", "unknown")),
                chunk_index=payload.get("chunk_index", -1),
                payload=payload,
            ))

        logger.debug(f"Search returned {len(hits)} hits (top_k={top_k})")
        return hits

    def search_with_context(
        self,
        query_vector: list[float],
        top_k: int = 5,
        max_context_chars: int = 4096,
    ) -> str:
        """
        搜索并拼接为 LLM 上下文文本

        Args:
            query_vector: 查询向量
            top_k: 检索数量
            max_context_chars: 最大上下文长度

        Returns:
            拼接好的上下文字符串
        """
        hits = self.search(query_vector, top_k=top_k)

        parts: list[str] = []
        total_chars = 0

        for hit in hits:
            entry = f"[来源: {hit.source} | 相似度: {hit.score:.4f}]\n{hit.text}"
            if total_chars + len(entry) > max_context_chars:
                break
            parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(parts) if parts else "(无检索结果)"
