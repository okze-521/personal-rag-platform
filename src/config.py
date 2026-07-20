"""
Pydantic-settings 配置管理
读取 .env 文件，提供类型安全的配置访问

使用方式:
    from src.config import settings
    print(settings.INFERENCE_HOST)   # "192.168.3.200"
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── .env 文件定位 ──────────────────────────────────
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """应用全局配置，所有字段均从 .env / 环境变量读取"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── 服务器 ────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ── 向量数据库 (Qdrant) ───────────────────────
    VECTOR_DB_HOST: str = "127.0.0.1"
    VECTOR_DB_PORT: int = 6333
    VECTOR_STORE_NAME: str = "okze_rag_collection"

    # ── Embedding 模型 ────────────────────────────
    EMBEDDING_MODEL: str = "bge-m3"
    EMBEDDING_DIM: int = 1024

    # ── 推理算力池 (台式机 Ollama) ────────────────
    INFERENCE_HOST: str = "192.168.3.200"
    INFERENCE_PORT: int = 11434
    MODEL_NAME: str = "qwen3.6:35b-a3b-q4_K_M"

    # ── 推理参数 ──────────────────────────────────
    CONTEXT_SIZE: int = 8192
    GENERATION_TIMEOUT: int = 60
    RETRIES: int = 3

    # ── CORS ──────────────────────────────────────
    CORS_ORIGINS: str = "*"

    # ── 计算属性 ──────────────────────────────────
    @property
    def ollama_api_base(self) -> str:
        return f"http://{self.INFERENCE_HOST}:{self.INFERENCE_PORT}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.VECTOR_DB_HOST}:{self.VECTOR_DB_PORT}"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


# ── 全局单例 ──────────────────────────────────────
settings = Settings()
