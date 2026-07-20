"""
文档导入 CLI — 命令行入口

用法:
    # 导入目录下所有 .txt/.md 文件
    python -m src.ingest docs/

    # 导入单个文件
    python -m src.ingest docs/0_架构设计.md

    # 指定集合名
    python -m src.ingest docs/ --collection my_collection

    # 切换切分策略
    python -m src.ingest docs/ --strategy sliding_window
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from src.ingest.document_loader import load_file, load_directory
from src.ingest.chunker import chunk_text
from src.ingest.embedder import OllamaEmbedder
from src.ingest.qdrant_writer import QdrantWriter

logger = logging.getLogger(__name__)


def ingest_path(
    target: str,
    collection_name: str | None = None,
    chunk_size: int = 500,
    strategy: str = "paragraph",
    force_recreate: bool = False,
) -> int:
    """
    导入文件或目录

    Returns:
        成功导入的文本块数量
    """
    target_path = Path(target)

    # ── 1. 加载文档 ────────────────────────────
    if target_path.is_file():
        docs = [load_file(target_path)]
        logger.info(f"Loading single file: {target_path.name}")
    elif target_path.is_dir():
        docs = load_directory(target_path)
        logger.info(f"Loading {len(docs)} files from: {target_path}")
    else:
        logger.error(f"Target not found: {target}")
        return 0

    if not docs:
        logger.warning("No documents to ingest")
        return 0

    # ── 2. 切分文档 ────────────────────────────
    all_chunks: list[tuple[str, str, int]] = []  # (filename, text, index)
    for filename, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, strategy=strategy)
        for i, chunk in enumerate(chunks):
            all_chunks.append((filename, chunk, i))
    logger.info(f"Chunked into {len(all_chunks)} chunks (strategy={strategy}, size={chunk_size})")

    # ── 3. 向量化 ──────────────────────────────
    embedder = OllamaEmbedder()
    texts = [c[1] for c in all_chunks]
    vectors = embedder.embed_batch(texts)

    # ── 4. 写入 Qdrant ─────────────────────────
    writer = QdrantWriter(collection_name=collection_name)

    if force_recreate:
        writer.ensure_collection()
        writer.delete_all()
    else:
        writer.ensure_collection()

    points = []
    for i, ((filename, chunk_content, chunk_idx), vec) in enumerate(zip(all_chunks, vectors)):
        if not vec:  # 跳过嵌入失败的块
            continue
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {
                "text": chunk_content,
                "source": filename,
                "chunk_index": chunk_idx,
            },
        })

    count = writer.upsert(points)
    return count


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="Personal RAG Platform — 文档导入工具",
    )
    parser.add_argument(
        "target",
        help="文件或目录路径 (支持 .txt / .md)",
    )
    parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Qdrant 集合名称 (默认: okze_rag_collection)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="每块最大字符数 (默认: 500)",
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=["paragraph", "sliding_window"],
        default="paragraph",
        help="切分策略 (默认: paragraph)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重建集合 (清空已有数据)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # ── 日志配置 ────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 执行导入 ────────────────────────────────
    count = ingest_path(
        target=args.target,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        strategy=args.strategy,
        force_recreate=args.force,
    )

    print(f"\n✅ 导入完成: {count} 条文档块")


if __name__ == "__main__":
    main()
