"""
LLM 客户端 — 推理请求封装 (本地 Ollama / 云端 Fallback)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from src.config import settings

logger = logging.getLogger(__name__)

# ── Fallback 配置 ──────────────────────────────
DEEPSEEK_API_KEY = settings.__dict__.get("DEEPSEEK_API_KEY", "")
CLOUD_LLM_URL = "https://api.deepseek.com/v1/chat/completions"


class LLMClient:
    """
    LLM 推理客户端 — 本地优先，云端降级

    使用方式:
        client = LLMClient()
        answer = client.generate(prompt="你好，请解释一下RAG")
    """

    def __init__(
        self,
        local_url: str | None = None,
        local_model: str | None = None,
        cloud_api_key: str | None = None,
    ) -> None:
        self.local_url = local_url or f"{settings.ollama_api_base}/api/chat"
        self.local_model = local_model or settings.MODEL_NAME
        self.cloud_api_key = cloud_api_key or DEEPSEEK_API_KEY
        self.timeout_local = settings.GENERATION_TIMEOUT

    def _try_local(self, prompt: str) -> str | None:
        """尝试调用本地台式机 Ollama"""
        try:
            resp = requests.post(
                self.local_url,
                json={
                    "model": self.local_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=self.timeout_local,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            logger.info(f"Local LLM success ({self.local_model})")
            return content
        except Exception as e:
            logger.warning(f"Local LLM unavailable: {type(e).__name__}: {e}")
            return None

    def _try_cloud(self, prompt: str) -> str | None:
        """调用云端 DeepSeek API"""
        if not self.cloud_api_key:
            logger.warning("No DEEPSEEK_API_KEY configured, skipping cloud fallback")
            return None

        try:
            resp = requests.post(
                CLOUD_LLM_URL,
                headers={"Authorization": f"Bearer {self.cloud_api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.info("Cloud LLM success (deepseek-chat)")
            return content
        except Exception as e:
            logger.error(f"Cloud LLM failed: {type(e).__name__}: {e}")
            return None

    def generate(self, prompt: str) -> str:
        """
        生成回答 — 本地优先，失败时自动降级到云端

        Returns:
            LLM 生成的回答文本，失败时返回错误信息
        """
        # 步骤 1: 本地台式机 Ollama
        result = self._try_local(prompt)
        if result:
            return result

        # 步骤 2: 云端 DeepSeek
        result = self._try_cloud(prompt)
        if result:
            return result

        return "⚠️ 生成失败: 本地与云端 LLM 均不可用，请检查网络连接和 API 配置。"
