"""MyAgent 的统一标准库日志初始化。"""

import logging
import sys
from typing import TextIO


_HANDLER_MARKER = "_myagent_console_handler"
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(
    debug: bool = False,
    *,
    stream: TextIO | None = None,
) -> None:
    """为根日志器配置唯一的 MyAgent 控制台处理器和日志级别。"""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in root_logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            handler.setLevel(logging.DEBUG if debug else logging.INFO)
            return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, _HANDLER_MARKER, True)
    root_logger.addHandler(handler)
