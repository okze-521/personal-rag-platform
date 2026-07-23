"""
BGE-Reranker 重排序模块

在向量检索后，用 Cross-Encoder 对 (问题, 文档) 逐对打分，
筛选出真正相关的文档再送 LLM，显著提高回答质量。

原理:
    向量检索 (Bi-Encoder) — 速度快但精度有限，靠余弦相似度
    重排序 (Cross-Encoder)   — 慢但精细，问题+文档拼接后联合编码打分

使用方式:
    from src.reranker import Reranker
    reranker = Reranker()
    top_docs = reranker.rerank("什么是存算分离？", docs, top_k=3)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from src.config import settings

if TYPE_CHECKING:
    from src.vectorstores.qdrant_store import SearchHit

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder 重排序器

    模型: BAAI/bge-reranker-v2-m3 (与 BGE-M3 Embedding 同系列，配合最佳)
    输入: (query, document) 文本对
    输出: 0~1 相关度分数，分数越高越相关
    """

    def __init__(self) -> None:
        model_name = settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {model_name} ...")
        # 优先读本地缓存，避免跨墙访问 huggingface.co
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu",
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            logger.warning("本地缓存不完整，尝试从镜像下载...")
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu",
                trust_remote_code=True,
            )
        logger.info("Reranker loaded ✓")

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """
        对检索结果重排序，返回最相关的 top_k 条

        Args:
            query:  用户问题
            hits:   向量检索返回的候选文档列表
            top_k:  最终保留数量 (默认使用 settings.RERANK_TOP_K)

        Returns:
            按相关度降序排列的文档列表
        """
        if not hits:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        # ── 1. 构造 (query, doc) 文本对 ──────────────
        pairs = [(query, hit.text) for hit in hits]

        # ── 2. Cross-Encoder 批量打分 ──────────────────
        scores = self.model.predict(pairs)  # type: ignore[arg-type]

        # ── 3. 按分数排序 ────────────────────────────
        scored = list(zip(hits, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = [hit for hit, _ in scored[:top_k]]

        # ── 4. 日志 ──────────────────────────────────
        logger.info(
            f"Reranked {len(hits)} → {len(reranked)} | "
            + " | ".join(
                f"#{i} score={score:.4f}"
                for i, (_, score) in enumerate(scored[:top_k], 1)
            )
        )

        return reranked
