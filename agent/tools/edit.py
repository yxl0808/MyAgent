# encoding:utf-8
"""
MyAgent 工具 — 文件编辑
=======================
通过精确字符串匹配替换文件内容。
这是最精细的文件修改工具——不改整行，只改匹配到的字符串。
"""

import logging
import os
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class EditTool(BaseTool):
    """
    在文件中查找指定字符串并替换为新的字符串。

    和 WriteTool 的区别:
      WriteTool: 覆盖整个文件（全量替换）
      EditTool:  只替换文件中匹配到的某一段文本（局部修改）

    替换规则:
      1. 精确匹配 old_string（包括空格、缩进、换行）
      2. 如果 old_string 在文件中出现多次：
         - replace_all=True  → 全部替换
         - replace_all=False → 报错，要求提供更具体的上下文
      3. 自动创建备份（.bak 文件），出错可恢复

    参数:
        path:        文件路径
        old_string:  要替换的原始文本
        new_string:  替换后的新文本
        replace_all: 是否替换所有匹配（默认 False，只替换第一次出现）
    """

    name = "edit"
    description = (
        "Make a precise string replacement in an existing file. "
        "Finds old_string in the file and replaces it with new_string. "
        "The old_string must match exactly (including spaces and line breaks). "
        "If the string appears multiple times, set replace_all=True to replace all, "
        "or provide more surrounding context to make the match unique."
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "Path to the file to edit",
            "required": True,
        },
        "old_string": {
            "type": "string",
            "description": "The exact text to find and replace (must be unique in the file unless replace_all=True)",
            "required": True,
        },
        "new_string": {
            "type": "string",
            "description": "The replacement text",
            "required": True,
        },
        "replace_all": {
            "type": "boolean",
            "description": "If True, replace all occurrences. If False (default), require a unique match.",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        执行字符串替换编辑。

        执行过程:
          1. 读取文件全部内容
          2. 统计 old_string 在文件中出现的次数
          3. 如果 0 次 → 返回错误
          4. 如果 >1 次且 replace_all=False → 返回错误并提供每个出现位置的上下文
          5. 执行替换
          6. 写入文件（先备份为 .bak）

        入参:
            params: {"path":"config.py", "old_string":"debug=False", "new_string":"debug=True"}

        返回:
            操作结果描述（包含替换次数、位置等信息）
        """
        path = params.get("path", "")
        old = params.get("old_string", "")
        new = params.get("new_string", "")
        replace_all = params.get("replace_all", False)

        # ── 校验 ──
        if not path:
            return "Error: No file path provided"
        path = self._resolve_path(path)
        if not old:
            return "Error: old_string cannot be empty"
        # old == new 是无操作，直接返回
        if old == new:
            return f"No change: old_string and new_string are identical in {path}"

        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if os.path.isdir(path):
            return f"Error: Path is a directory: {path}"

        # ── 读取文件 ──
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        # ── 统计匹配次数 ──
        count = original.count(old)

        if count == 0:
            # 找不到匹配 → 尝试给 LLM 有用的提示
            # 显示文件中包含 old_string 前 20 个字符的内容片段
            snippet = old[:20]
            hint_lines = []
            for i, line in enumerate(original.splitlines(), 1):
                if snippet in line:
                    hint_lines.append(f"  Line {i}: {line.strip()[:80]}")
            hint = ""
            if hint_lines:
                hint = "\nDid you mean one of these lines?\n" + "\n".join(hint_lines[:5])
            return f"Error: old_string not found in {path} (searched for {len(old)} chars){hint}"

        if count > 1 and not replace_all:
            # 不唯一 → 列出每个匹配的位置，帮助 LLM 提供更多上下文
            positions = self._find_all_positions(original, old)
            pos_info = []
            for i, pos in enumerate(positions[:10]):  # 最多显示 10 个
                # 计算行号
                line_num = original[:pos].count("\n") + 1
                # 显示匹配处前后各 40 个字符的上下文
                start = max(0, pos - 20)
                end = min(len(original), pos + len(old) + 20)
                ctx = original[start:end].replace("\n", "\\n")
                pos_info.append(f"  Match {i + 1} at line {line_num}: ...{ctx}...")

            info = "\n".join(pos_info)
            return (
                f"Error: old_string found {count} times in {path}. "
                f"Provide more surrounding context to make the match unique, "
                f"or set replace_all=True to replace all occurrences.\n"
                f"Matches:\n{info}"
            )

        # ── 执行替换 ──
        try:
            # 先备份
            backup_path = path + ".bak"
            with open(backup_path, mode="w", encoding="utf-8") as f:
                f.write(original)

            # 替换
            if replace_all:
                modified = original.replace(old, new)
            else:
                modified = original.replace(old, new, 1)

            # 写入
            with open(path, mode="w", encoding="utf-8") as f:
                f.write(modified)

            logger.info("[Edit] Replaced %d occurrence(s) in %s", count if replace_all else 1, path)
            return (
                f"Successfully edited {path}\n"
                f"  Replaced {count if replace_all else 1} occurrence(s)\n"
                f"  Backup saved to {backup_path}"
            )

        except Exception as e:
            logger.error("[Edit] Failed: %s", e)
            return f"Error editing file: {e}"

    @staticmethod
    def _find_all_positions(text: str, substring: str) -> list:
        """
        查找 substring 在 text 中的所有起始位置。

        入参:
            text: 完整文本
            substring: 要搜索的子串

        返回:
            起始索引列表，如 [42, 156, 230]
        """
        positions = []
        start = 0
        while True:
            pos = text.find(substring, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions