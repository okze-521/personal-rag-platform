"""
文档加载器 — 读取文件内容并剥离 YAML frontmatter (Obsidian 元数据)
"""
from __future__ import annotations

from pathlib import Path


def strip_frontmatter(text: str) -> str:
    """去掉 Markdown 文件开头的 YAML frontmatter (---...---)"""
    text = text.strip()
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].strip()
    return text


def load_file(filepath: str | Path) -> tuple[str, str]:
    """
    读取单个文件，返回 (文件名, 清洗后文本)

    支持: .txt, .md, .markdown
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    text = strip_frontmatter(raw) if filepath.suffix in (".md", ".markdown") else raw
    return filepath.name, text


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


def load_directory(dirpath: str | Path, extensions: set[str] | None = None) -> list[tuple[str, str]]:
    """
    批量加载目录下所有匹配文件

    返回: [(文件名, 清洗后文本), ...]
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"不是有效目录: {dirpath}")

    exts = extensions or SUPPORTED_EXTENSIONS

    docs = []
    for filepath in sorted(dirpath.iterdir()):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in exts:
            continue
        try:
            name, text = load_file(filepath)
            docs.append((name, text))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"跳过 {filepath.name}: {e}")

    return docs
