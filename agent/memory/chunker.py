# encoding:utf-8
"""
文本按行分块。markdown 以换行为自然语义边界，不需要递归字符分块。
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TextChunk:
    """一块文本 + 在原文件中的行号范围"""
    text: str
    start_line: int
    end_line: int


class TextChunker:
    """按行分块，超 max_tokens 时切新块，块间有重叠。"""

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def _chars_per_token(text: str) -> float:
        """根据中英文字符比例动态估算 chars/token。

        中文 ≈ 0.67 chars/token（1 个中文字约 1.5 token）
        英文 ≈ 4 chars/token（4 个英文字母约 1 token）
        混合文本按比例加权。
        """
        if not text:
            return 4.0
        cjk = sum(1 for c in text if ord(c) > 127)
        total = len(text)
        ratio = cjk / total
        return ratio * 0.67 + (1 - ratio) * 4.0

    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        将文本按行拆分为 chunk 列表。

        一块一块地累积行，超上限就切，然后回退几行做重叠。
        """
        if not text.strip():
            return []

        lines = text.split("\n")
        cpt = self._chars_per_token(text)
        max_chars = int(self.max_tokens * cpt)
        overlap_chars = int(self.overlap_tokens * cpt)

        chunks: List[TextChunk] = []
        i = 0

        while i < len(lines):
            chunk_lines: List[str] = []
            chunk_chars = 0
            start = i + 1  # 当前块起始行号（1-based）

            # 攒行，直到塞不下
            while i < len(lines):
                line = lines[i]
                if chunk_chars + len(line) > max_chars and chunk_lines:
                    break  # 满了，切
                chunk_lines.append(line)
                chunk_chars += len(line)
                i += 1

            # 如果一行都没攒下（单行本身超长），硬切这一行
            if not chunk_lines and i < len(lines):
                line = lines[i]
                for j in range(0, len(line), max_chars):
                    chunks.append(TextChunk(
                        text=line[j:j + max_chars],
                        start_line=i + 1, end_line=i + 1,
                    ))
                i += 1
                continue

            end = start + len(chunk_lines) - 1
            chunks.append(TextChunk(
                text="\n".join(chunk_lines),
                start_line=start, end_line=end,
            ))

            # 还没到末尾 → 回退末尾几行作为下一块的开头（重叠）
            if i < len(lines):
                back = 0
                back_lines = 0
                for line in reversed(chunk_lines):
                    if back + len(line) > overlap_chars:
                        break
                    back += len(line)
                    back_lines += 1
                i -= back_lines

        return chunks
