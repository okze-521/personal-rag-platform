"""
Pydantic 数据模型 (DTO)
定义 FastAPI 请求/响应体结构

使用方式:
    from src.models import QueryRequest, QueryResponse, HealthResponse
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """存活探针响应"""
    status: str = "ok"


class QueryRequest(BaseModel):
    """RAG 查询请求体"""
    query: str = Field(
        ...,
        description="用户输入的查询文本",
        min_length=1,
        max_length=4096,
    )
    filter_metadata: dict | None = Field(
        default=None,
        description="可选元数据过滤 (部门/类型/日期范围)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="检索返回的 Top-K 文档数量",
    )


class ContextSnippet(BaseModel):
    """单条检索上下文"""
    source: str = Field(description="来源文档标识")
    text: str = Field(description="文本片段内容")
    score: float = Field(description="相似度分数 (0~1)")


class QueryResponse(BaseModel):
    """RAG 查询响应体"""
    original_query: str = Field(description="原始查询文本")
    context_snippets: list[str] = Field(description="检索到的上下文片段")
    generated_response: str = Field(description="LLM 生成的最终回答")
