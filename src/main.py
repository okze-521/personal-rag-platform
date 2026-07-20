"""
personal_rag_platform — FastAPI Service Entry Point
Author: Okze | Architecture: Storage-Compute Separated

启动命令:
    uv run python src/main.py
或
    python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── 项目根目录注入 ────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 加载 .env ──────────────────────────────────
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ENV_FILE)

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models import QueryRequest, QueryResponse, HealthResponse

# ── Application ────────────────────────────────
app = FastAPI(
    title="Personal RAG Platform",
    version="2.0.0",
    description="存算分离 · Qdrant + Ollama MoE 混合算力推理",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 路由 ───────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """存活探针"""
    return HealthResponse(status="ok")


@app.post("/api/v1/query", tags=["rag"])
async def query(body: QueryRequest) -> QueryResponse:
    """RAG 检索增强生成入口: Embed → Retrieve → Generate"""
    try:
        from src.rag_chain import get_rag_chain

        chain = await get_rag_chain()
        result = await chain.ainvoke({"input": body.query})

        return QueryResponse(
            original_query=body.query,
            context_snippets=result.get("context", []),
            generated_response=result.get("output", "⚠️ 模型未返回有效响应"),
        )
    except Exception as exc:
        from src.utils.logger_config import get_logger
        logger = get_logger(__name__)
        logger.error(f"Pipeline Error: {exc}", exc_info=True)
        return QueryResponse(
            original_query=body.query,
            context_snippets=[],
            generated_response=f"Error [{type(exc).__name__}]: {exc}",
        )


# ── Main ───────────────────────────────────────

def main():
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, reload=True)


if __name__ == "__main__":
    main()
