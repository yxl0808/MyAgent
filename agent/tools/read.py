# encoding:utf-8
"""
MyAgent 工具 — 文件读取
=======================
读取文件内容并返回。支持分页（offset + limit），大文件也能安全读取。
"""

import logging
import os
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ReadTool(BaseTool):
    """
    读取文件内容，带行号。

    支持分段读取:
      - offset: 从第几行开始读（默认 0，即第一行）
      - limit:  最多读多少行（默认 2000，防止一次返回太多内容撑爆上下文）

    参数:
        path:   文件路径
        offset: 起始行号（0-indexed，即 0 = 第一行）
        limit:  最多读取的行数（默认 2000）
    """

    name = "read"
    description = (
        "Read content of a file with line numbers. "
        "Use this to examine source code, configuration files, or any text file. "
        "For large files, use offset and limit to read specific sections."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to the file to read",
            "required": True,
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (0 = first line). Default 0.",
            "required": False,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of lines to read. Default 2000.",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        读取文件并返回带行号的内容。

        输出格式:
            1  | import os
            2  | import sys
            3  |
            ...

        入参:
            params: {"path": "config.py", "offset": 0, "limit": 50}

        返回:
            带行号的文件内容，超过限制时提示截断信息
        """
        path = params.get("path", "")
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 2000))

        # 校验路径
        if not path:
            return "Error: No file path provided"
        path = self._resolve_path(path)

        # 检查文件
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return f"Error: Path is a directory, not a file: {path}. Use ls tool to list directory contents."

        try:
            with open(path, mode="r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except UnicodeDecodeError:
            # 可能是二进制文件，尝试用 latin-1 读取
            try:
                with open(path, mode="r", encoding="latin-1", errors="replace") as f:
                    all_lines = f.readlines()
            except Exception as e:
                return f"Error: Cannot read file (possibly binary): {e}"
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

        total_lines = len(all_lines)

        # 分段
        end = offset + limit
        selected = all_lines[offset:end]

        # 格式化输出：每行前加行号
        lines = []
        for i, line in enumerate(selected):
            line_num = offset + i + 1  # 行号从 1 开始显示
            # 去掉行尾的换行符再拼回去，保持输出整洁
            lines.append(f"{line_num:<6}| {line.rstrip()}")

        output = "\n".join(lines)

        # 头部信息
        header = f"File: {path}  (lines {offset + 1}-{offset + len(selected)} of {total_lines})\n"
        header += "-" * 70

        # 尾部提示
        footer = ""
        if end < total_lines:
            footer = f"\n\n... (truncated, {total_lines - end} more lines. Use offset={end} to continue reading)"

        return header + "\n" + output + footer