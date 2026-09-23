# encoding:utf-8
"""
MyAgent 工具 — 文件写入
=======================
创建新文件或覆盖已有文件的内容。
"""

import logging
import os
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WriteTool(BaseTool):
    """
    将指定内容写入文件。如果文件已存在则覆盖。

    使用场景:
      - 创建新的代码文件
      - 修改已有文件的全部内容
      - 写入配置文件

    注意:
      - 此工具是覆盖写入，不是追加
      - 如需局部修改，使用 EditTool（精确字符串替换）
      - 会自动创建不存在的父目录

    参数:
        path:    文件路径
        content: 要写入的文本内容
    """

    name = "write"
    description = (
        "Write content to a file, creating it if it doesn't exist or overwriting if it does. "
        "Parent directories are created automatically if needed. "
        "Use this to create new files or completely replace file contents. "
        "For targeted edits to existing files, use the edit tool instead."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to the file to write",
            "required": True,
        },
        "content": {
            "type": "string",
            "description": "The text content to write to the file",
            "required": True,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        将 content 写入 path 指定的文件。

        执行过程:
          1. 检查路径是否为空
          2. 如果父目录不存在，自动创建（makedirs）
          3. 以 utf-8 编码写入文件
          4. 返回成功确认信息（含文件大小和写入行数）

        入参:
            params: {"path": "hello.py", "content": "print('hello')"}

        返回:
            操作结果描述字符串
        """
        path = params.get("path", "")
        content = params.get("content", "")

        if not path:
            return "Error: No file path provided"
        path = self._resolve_path(path)

        try:
            # 确保父目录存在
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                logger.info("[Write] Created parent directory: %s", parent_dir)

            # 记录是新建还是覆盖
            is_new = not os.path.exists(path)

            with open(path, mode="w", encoding="utf-8") as f:
                f.write(content)

            # 统计写入信息
            file_size = os.path.getsize(path)
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            action = "Created" if is_new else "Updated"

            logger.info("[Write] %s file: %s (%d lines, %d bytes)", action, path, line_count, file_size)
            return f"{action} file: {path}\n  Lines: {line_count}\n  Size: {file_size} bytes"

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            logger.error("[Write] Failed: %s", e)
            return f"Error writing file: {e}"