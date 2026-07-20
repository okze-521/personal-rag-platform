"""
Qdrant 写入器 — 向量数据批量写入 Qdrant 集合
"""
from __future__ import annotations

import logging
import uuid

import requests

from src.config import settings

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────
VECTOR_DIM = 1024         # BGE-M3 向量维度
DISTANCE_METRIC = "Cosine"


class QdrantWriter:
    """
    Qdrant 文档写入封装

    使用方式:
        writer = QdrantWriter()
        writer.ensure_collection()
        writer.upsert(points)
    """

    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.qdrant_url
        self.collection_name = collection_name or settings.VECTOR_STORE_NAME

    @property
    def _collection_url(self) -> str:
        return f"{self.base_url}/collections/{self.collection_name}"

    @property
    def _points_url(self) -> str:
        return f"{self._collection_url}/points"

    def collection_exists(self) -> bool:
        """检查集合是否存在"""
        resp = requests.get(
            f"{self.base_url}/collections/{self.collection_name}",
            timeout=10,
        )
        return resp.status_code == 200

    def ensure_collection(self, vector_dim: int = VECTOR_DIM) -> bool:
        """
        确保集合存在，不存在则自动创建

        返回: True=新创建, False=已存在
        """
        if self.collection_exists():
            logger.info(f"Collection '{self.collection_name}' already exists")
            return False

        resp = requests.put(
            f"{self.base_url}/collections/{self.collection_name}",
            json={
                "vectors": {
                    "size": vector_dim,
                    "distance": DISTANCE_METRIC,
                },
            },
            timeout=10,
        )
        resp.raise_for_status()
        logger.info(f"Created collection '{self.collection_name}' | dim={vector_dim} | metric={DISTANCE_METRIC}")
        return True

    def upsert(self, points: list[dict]) -> int:
        """
        批量插入 / 更新向量点到 Qdrant

        Args:
            points: [{"id": str, "vector": [...], "payload": {...}}, ...]

        Returns:
            成功写入的点数
        """
        if not points:
            return 0

        resp = requests.put(
            f"{self._points_url}?wait=true",
            json={"points": points},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        status = result.get("result", {}).get("status", "unknown")
        logger.info(f"Upserted {len(points)} points → status={status}")
        return len(points)

    def delete_all(self) -> None:
        """清空集合所有数据"""
        resp = requests.post(
            f"{self._points_url}/delete",
            json={"filter": {}},  # 空 filter = 全部删除
            timeout=30,
        )
        resp.raise_for_status()
        logger.warning(f"Deleted ALL points from '{self.collection_name}'")
