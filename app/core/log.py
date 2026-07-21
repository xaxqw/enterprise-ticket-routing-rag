"""统一日志入口。

替代散落在各处的 print，统一走标准库 logging。
所有模块用 `from app.core.log import get_logger; logger = get_logger(__name__)` 获取 logger。
"""
import logging
import sys

_configured = False


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _ensure_configured()
    return logging.getLogger(name)
