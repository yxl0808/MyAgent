"""MyAgent 的控制台文本输出工具。"""

import sys
from typing import TextIO


def print_safely(value: object, *, stream: TextIO | None = None) -> None:
    """输出文本，并替换目标编码不支持的字符。"""
    target_stream = stream or sys.stdout
    text = str(value)
    encoding = getattr(target_stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="replace").decode(encoding)
    print(text, file=target_stream)
