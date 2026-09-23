# encoding:utf-8
"""
对话摘要与记忆蒸馏。

Flush: 对话消息 -> 摘要 -> 追加到 memory/YYYY-MM-DD.md
Refine: 近期日记 + MEMORY.md -> 长期记忆蒸馏
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

FLUSH_SYSTEM_ZH = """你是一个对话记录助手。请将对话内容归纳为当天的日常记录。

## 要求
按「事件」维度归纳发生的事，不要按对话轮次逐条记录：
- 每条一行，用 "- " 开头
- 合并同一件事的多轮对话
- 只记录有意义的事件，忽略闲聊和问候
- 保留关键的决策、结论和待办事项

当对话没有任何记录价值（仅含问候或无意义内容），直接回复"无"。"""

FLUSH_SYSTEM_EN = """You are a conversation-logging assistant. Summarize the conversation into a daily record.

## Requirements
Summarize by event, not turn by turn:
- One item per line, starting with "- "
- Merge multiple turns about the same thing
- Only record meaningful events; ignore small talk and greetings
- Keep key decisions, conclusions and to-dos

If the conversation has no record value, reply with exactly "None"."""

FLUSH_USER_ZH = "请归纳以下对话的日常记录：\n\n{conversation}"
FLUSH_USER_EN = "Summarize the daily record of the following conversation:\n\n{conversation}"

REFINE_SYSTEM_ZH = """你是一个记忆整理助手，负责定期整理用户的长期记忆。

你将收到两份材料：
1. 当前长期记忆：MEMORY.md 的全部现有内容
2. 近期日记：最近几天的日常记录

MEMORY.md 会注入每次对话的系统提示词中，因此必须保持精炼，只存放有价值和值得记忆的内容。

重要：只能基于提供的材料进行整理，严禁编造、推测或添加材料中不存在的信息。

请输出完整的更新后 MEMORY.md 内容：
- 合并含义相近的条目
- 提取近期日记中值得长期保存的信息
- 新旧信息冲突时，以近期日记为准
- 删除临时性记录、重复内容和空白条目
- 每条一行，用 "- " 开头
- 可用 "## 标题" 对相关条目分组
- 尽量控制在 50 条以内"""

REFINE_SYSTEM_EN = """You are a memory-curation assistant that periodically organizes the user's long-term memory.

You will receive two inputs:
1. Current long-term memory: the full existing MEMORY.md content
2. Recent diary: recent daily records

MEMORY.md is injected into the system prompt of each conversation, so it must stay concise and valuable.

Important: organize strictly based on the provided material. Never fabricate or infer information not present in it.

Output the complete updated MEMORY.md content:
- Merge semantically similar items
- Extract long-term useful information from recent diary
- Prefer recent diary when old and new information conflict
- Remove temporary notes, duplicates and blank items
- One item per line, starting with "- "
- You may group related items under "## headings"
- Keep it under about 50 items"""

REFINE_USER_ZH = "## 当前长期记忆\n\n{memory_content}\n\n## 近期日记（最近 {days} 天）\n\n{daily_content}"
REFINE_USER_EN = "## Current long-term memory\n\n{memory_content}\n\n## Recent diary (last {days} days)\n\n{daily_content}"


def _is_zh() -> bool:
    """检测当前界面语言是否应使用中文提示词。"""
    try:
        from config import conf
        return conf().get("language", "auto") != "en"
    except Exception:
        return True


def _is_empty(raw: str) -> bool:
    """判断 LLM 摘要结果是否表示无记录价值。"""
    s = raw.strip()
    return s in ("", "无", "None", "none")


class MemoryFlushManager:
    """管理对话摘要写入和长期记忆蒸馏。"""

    def __init__(self, workspace_dir: Path, llm_model: Any = None):
        """初始化摘要管理器。"""
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.llm_model = llm_model
        self.memory_dir = self.workspace_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._last_refine_hash = ""

    def _get_main_memory(self) -> Path:
        """返回主长期记忆文件路径。"""
        return self.workspace_dir / "MEMORY.md"

    def _get_today_file(self) -> Path:
        """返回今日记忆文件路径；不存在时自动创建。"""
        today = datetime.now().strftime("%Y-%m-%d")
        fpath = self.memory_dir / f"{today}.md"
        if not fpath.exists():
            fpath.write_text(f"# Daily Memory: {today}\n\n", encoding="utf-8")
        return fpath

    @staticmethod
    def _extract_text(content: Any) -> str:
        """从消息 content 字段中提取纯文本。"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    def _format_conversation(self, messages: List[Dict], max_messages: int = 0) -> str:
        """把消息列表格式化为可读的对话文本。"""
        msgs = messages if max_messages == 0 else messages[-max_messages * 2:]
        lines = []
        for msg in msgs:
            role = msg.get("role", "")
            text = self._extract_text(msg.get("content", ""))
            if not text.strip():
                continue
            if role == "user":
                lines.append(f"用户: {text[:500]}")
            elif role == "assistant":
                lines.append(f"助手: {text[:500]}")
        return "\n".join(lines)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 做摘要或蒸馏，返回原始文本输出。"""
        if not self.llm_model:
            raise RuntimeError("No LLM model available for summarization")

        response = self.llm_model.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        return response.get("text", "") if isinstance(response, dict) else str(response)

    def _fallback_summary(self, messages: List[Dict], max_messages: int) -> str:
        """LLM 不可用时的规则摘要兜底。"""
        msgs = messages[-max_messages * 2:] if max_messages else messages
        events = []
        user_text = ""
        for msg in msgs:
            role = msg.get("role", "")
            text = self._extract_text(msg.get("content", "")).strip()
            if not text:
                continue
            if role == "user":
                user_text = text[:200]
            elif role == "assistant" and user_text:
                events.append(f"- 用户: {user_text} -> 回复: {text[:200]}")
                user_text = ""
        return "\n".join(events[:10]) if events else ""

    def flush_from_messages(
        self,
        messages: List[Dict],
        reason: str = "threshold",
        max_messages: int = 20,
    ) -> bool:
        """把对话消息摘要追加到今日记忆文件。"""
        conversation = self._format_conversation(messages, max_messages)
        if not conversation.strip():
            return False

        try:
            use_zh = _is_zh()
            system_prompt = FLUSH_SYSTEM_ZH if use_zh else FLUSH_SYSTEM_EN
            user_template = FLUSH_USER_ZH if use_zh else FLUSH_USER_EN
            raw = self._call_llm(
                system_prompt,
                user_template.format(conversation=conversation),
            )
        except Exception as e:
            logger.warning("[MemoryFlush] LLM summarization failed: %s", e)
            raw = self._fallback_summary(messages, max_messages)

        if _is_empty(raw):
            return False

        today_file = self._get_today_file()
        now_str = datetime.now().strftime("%H:%M")
        headers = {
            "threshold": f"## Trimmed ({now_str})",
            "overflow": f"## Overflow ({now_str})",
            "daily": f"## Daily ({now_str})",
            "trim": f"## Trimmed ({now_str})",
        }
        header = headers.get(reason, f"## Note ({now_str})")

        with open(today_file, "a", encoding="utf-8") as f:
            f.write(f"\n{header}\n\n{raw.strip()}\n")

        logger.info("[MemoryFlush] Wrote to %s (reason=%s)", today_file.name, reason)
        return True

    def refine(self, lookback_days: int = 7, force: bool = False) -> bool:
        """把近期 daily memory 蒸馏进主长期记忆 MEMORY.md。"""
        if not self.llm_model:
            logger.warning("[Refine] No LLM model, skipping")
            return False

        mem_file = self._get_main_memory()
        memory_content = (
            mem_file.read_text(encoding="utf-8").strip()
            if mem_file.exists()
            else ""
        )

        parts = []
        today = datetime.now().date()
        for offset in range(lookback_days):
            day = today - timedelta(days=offset)
            date_str = day.strftime("%Y-%m-%d")
            daily_file = self.memory_dir / f"{date_str}.md"
            if not daily_file.exists():
                continue
            content = daily_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"### {date_str}\n\n{content}")

        if not parts:
            logger.info("[Refine] No daily records, skipping")
            return False

        daily_content = "\n\n".join(parts)
        new_hash = hashlib.md5(
            (memory_content + daily_content).encode("utf-8")
        ).hexdigest()
        if not force and new_hash == self._last_refine_hash:
            logger.info("[Refine] No new content, skipping")
            return False

        try:
            use_zh = _is_zh()
            system_prompt = REFINE_SYSTEM_ZH if use_zh else REFINE_SYSTEM_EN
            user_template = REFINE_USER_ZH if use_zh else REFINE_USER_EN
            new_memory = self._call_llm(
                system_prompt,
                user_template.format(
                    memory_content=memory_content or "(empty)",
                    days=lookback_days,
                    daily_content=daily_content,
                ),
            ).strip()
        except Exception as e:
            logger.warning("[Refine] LLM call failed: %s", e)
            return False

        if not new_memory:
            return False

        mem_file.write_text(new_memory + "\n", encoding="utf-8")
        self._last_refine_hash = new_hash
        logger.info(
            "[Refine] MEMORY.md updated (%d -> %d chars)",
            len(memory_content),
            len(new_memory),
        )
        return True
