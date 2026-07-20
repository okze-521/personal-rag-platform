"""
LangChain RAG Pipeline — 检索增强生成核心链路

链路:
    Query → BGE-M3 Embedding → Qdrant 检索 → 上下文拼接 → LLM 推理

使用方式:
    from src.rag_chain import RAGPipeline, get_rag_chain
    pipeline = await get_rag_chain()
    result = await pipeline.ainvoke({"input": "什么是存算分离？"})
"""
from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.ingest.embedder import OllamaEmbedder
from src.vectorstores.qdrant_store import QdrantStore
from src.utils.ollama_client import LLMClient

logger = logging.getLogger(__name__)

# ── Prompt 模板 ────────────────────────────────
RAG_PROMPT = """你是一个基于私有知识库的智能助手。请根据以下参考资料详细回答问题。

## 参考资料：
{context}

## 问题：
{question}

## 要求：
- 基于参考资料作答，引用关键信息
- 如果资料不足以回答问题，请诚实说明
- 用中文回答，保持专业、清晰的表达

## 回答："""


class RAGPipeline:
    """
    RAG 核心管道

    组件:
        - OllamaEmbedder: BGE-M3 文本向量化
        - QdrantStore:   向量相似度检索
        - LLMClient:     Ollama/DeepSeek 推理 (本地优先)
    """

    def __init__(self) -> None:
        self.embedder = OllamaEmbedder()
        self.store = QdrantStore()
        self.llm = LLMClient()

        logger.info(
            f"RAGPipeline initialized | "
            f"embedder={settings.EMBEDDING_MODEL} | "
            f"qdrant={settings.qdrant_url} | "
            f"inference={settings.ollama_api_base}/{settings.MODEL_NAME}"
        )

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        执行一次完整的 RAG 检索增强生成

        Args:
            inputs: {"input": "查询问题", "top_k": 5 (可选)}

        Returns:
            {"input": ..., "context": [...], "output": "生成的回答"}
        """
        query = inputs.get("input", "")
        top_k = inputs.get("top_k", 5)

        if not query.strip():
            return {"input": query, "context": [], "output": "请输入有效问题。"}

        logger.info(f"RAG query | len={len(query)} | top_k={top_k}")

        try:
            # ── 1. Embedding ────────────────────────
            query_vec = self.embedder.embed(query)
            logger.debug(f"Embedded query → dim={len(query_vec)}")

            # ── 2. Qdrant 检索 ──────────────────────
            hits = self.store.search(query_vec, top_k=top_k)
            context = self.store.search_with_context(query_vec, top_k=top_k)
            context_snippets = [h.text for h in hits]

            if not hits:
                return {
                    "input": query,
                    "context": [],
                    "output": "⚠️ 未在知识库中检索到相关内容。请先导入文档 (python -m src.ingest docs/)。",
                }

            logger.info(f"Retrieved {len(hits)} hits | top_score={hits[0].score:.4f}")

            # ── 3. Prompt 拼装 ──────────────────────
            prompt = RAG_PROMPT.format(context=context, question=query)

            # ── 4. LLM 推理 ─────────────────────────
            answer = self.llm.generate(prompt)

            return {
                "input": query,
                "context": context_snippets,
                "output": answer,
            }

        except Exception as exc:
            logger.error(f"RAG pipeline error: {exc}", exc_info=True)
            return {
                "input": query,
                "context": [],
                "output": f"❌ 检索生成失败: {type(exc).__name__}: {exc}",
            }


# ── 全局单例 ──────────────────────────────────────────
_pipeline: RAGPipeline | None = None


async def get_rag_chain() -> RAGPipeline:
    """获取 RAG Pipeline 单例 (延迟初始化)"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
