"""
文本切分器 — 将长文档切分为适合 Embedding 的文本块

策略:
    1. 按段落分割 (\\n\\n)
    2. 超长段落按字符数强制截断
    3. 可选 token 级别切分 (需 tiktoken)
"""
from __future__ import annotations

import re
from typing import Iterator


# ── 配置 ──────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500   # 每块最大字符数
DEFAULT_OVERLAP = 50        # 块间重叠字符数


def chunk_by_paragraph(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    按段落切分，超长段落按 max_chars 截断
    """
    # 按双换行 / 单换行分割段落
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 超长段落: 按 max_chars 强制切分
            for i in range(0, len(para), max_chars):
                chunk = para[i : i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)

    return chunks


def chunk_by_sliding_window(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    滑窗切分 (适合结构化程度低的文本)
    """
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    strategy: str = "paragraph",
) -> list[str]:
    """
    通用切分入口

    Args:
        text: 待切分文本
        chunk_size: 每块最大字符数
        overlap: 块间重叠 (仅 sliding_window 策略)
        strategy: "paragraph" | "sliding_window"

    Returns:
        切分后的文本块列表
    """
    if strategy == "sliding_window":
        return chunk_by_sliding_window(text, chunk_size, overlap)
    return chunk_by_paragraph(text, chunk_size)
