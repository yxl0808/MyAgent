# encoding:utf-8
"""
MyAgent 工具 — 目录列表
=======================
列出指定目录下的文件和子目录。
Agent 用它来了解项目结构、确认文件是否存在、查看文件大小等。
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class LsTool(BaseTool):
    """
    列出目录内容，包含文件名、类型（文件/目录）、大小、修改时间。

    参数:
        path:     要列出的目录路径，默认 "."（当前目录）
        pattern:  可选的文件名过滤模式（简单字符串匹配），如 "*.py"
    """

    name = "ls"
    description = (
        "List files and directories in a given path. "
        "Shows name, type (file/dir), size, and last modified time. "
        "Use this to explore the project structure or verify file existence."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Directory path to list. Defaults to current directory '.'",
            "required": False,
        },
        "pattern": {
            "type": "string",
            "description": "Optional filename pattern to filter results, e.g. '*.py'",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        列出目录内容。

        输出格式:
            drwx  config.py       12.3 KB  2024-01-15 10:30
            -rwx  main.py          2.1 KB  2024-01-14 08:00

        入参:
            params: {"path": ".", "pattern": "*.py"}  (pattern 可选)

        返回:
            格式化的文件列表文本，或错误信息
        """
        path = params.get("path", ".") or "."
        pattern = params.get("pattern", "")
        path = self._resolve_path(path)

        # 检查路径是否存在
        if not os.path.exists(path):
            return f"Error: Path does not exist: {path}"
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"

        try:
            entries = os.listdir(path)
        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading directory: {e}"

        # 按名称排序（目录优先，然后文件）
        entries.sort(key=lambda name: (not os.path.isdir(os.path.join(path, name)), name.lower()))

        lines = []
        file_count = 0
        dir_count = 0

        for name in entries:
            # 过滤：如果指定了 pattern，只匹配符合的
            if pattern and not self._match_pattern(name, pattern):
                continue

            full_path = os.path.join(path, name)
            try:
                stat = os.stat(full_path)
            except OSError:
                continue

            is_dir = stat.st_mode & 0o40000  # 目录标志位
            # 类型 + 权限简写
            type_char = "d" if is_dir else "-"
            perm = self._mode_to_str(stat.st_mode)
            # 文件大小（人类可读）
            size_str = self._format_size(stat.st_size)
            # 修改时间
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            lines.append(f"{type_char}{perm}  {name:<30s}  {size_str:>8s}  {mtime}")

            if is_dir:
                dir_count += 1
            else:
                file_count += 1

        # 摘要
        summary = f"\n{dir_count} directories, {file_count} files"
        header = f"Directory: {path}\n" + "-" * 70

        return header + "\n" + ("\n".join(lines) if lines else "(empty)") + summary

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _match_pattern(name: str, pattern: str) -> bool:
        """通配符匹配，由 Python 标准库 fnmatch 处理。"""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)

    @staticmethod
    def _mode_to_str(mode: int) -> str:
        """将 st_mode 转为简单的 rwx 权限字符串（owner 部分）。"""
        chars = []
        for shift, char in [(6, 'r'), (5, 'w'), (4, 'x')]:
            chars.append(char if mode & (1 << shift) else '-')
        return ''.join(chars)

    @staticmethod
    def _format_size(size: int) -> str:
        """将字节数转为人类可读格式。"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}" if unit != 'B' else f"{size} B"
            size /= 1024
        return f"{size:.1f} TB"