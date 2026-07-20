"""
应用日志配置模块
提供统一的日志记录器，后续可接入 Sentry / Prometheus

使用方式:
    from src.utils.logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")
"""
from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
    return logger
