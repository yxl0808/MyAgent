# encoding:utf-8
"""
MyAgent 工具 — 记忆检索
=======================
让 Agent 能够检索和写入长期记忆。
目前是占位实现，等记忆系统模块完成后补全。
"""

import logging
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MemoryFlushTool(BaseTool):
    """
    手动把当前 Agent 对话历史摘要写入 daily memory。
    """

    name = "memory_flush"
    description = (
        "Flush the current conversation history into daily long-term memory. "
        "Use this when the conversation contains information that should be "
        "saved before context trimming or session handoff."
    )
    parameters = {
        "reason": {
            "type": "string",
            "description": "Flush reason, such as manual, threshold, overflow, daily, or trim.",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """把当前 Agent 对话历史写入 daily memory。"""
        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        messages = getattr(agent, "messages", [])
        if not messages:
            return "No conversation messages to flush."

        reason = str(params.get("reason", "manual") or "manual")
        ok = memory_manager.flush(messages, reason=reason)
        if not ok:
            return "No daily memory was written."

        return f"Flushed {len(messages)} conversation messages into daily memory."


class MemoryRefineTool(BaseTool):
    """
    手动把 recent daily memory 蒸馏进 MEMORY.md。
    """

    name = "memory_refine"
    description = (
        "Refine recent daily memory files into MEMORY.md. "
        "Use this to distill detailed daily notes into concise long-term memory."
    )
    parameters = {
        "lookback_days": {
            "type": "integer",
            "description": "How many recent daily memory files to include. Defaults to 7.",
            "required": False,
        },
        "force": {
            "type": "boolean",
            "description": "Whether to refine even when the input appears unchanged.",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """把 daily memory 蒸馏进 MEMORY.md。"""
        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        lookback_days = int(params.get("lookback_days", 7) or 7)
        lookback_days = max(1, lookback_days)
        force = params.get("force") is True
        ok = memory_manager.refine(lookback_days=lookback_days, force=force)
        if not ok:
            return "MEMORY.md was not updated."

        return "Refined daily memory into MEMORY.md."


class MemorySearchTool(BaseTool):
    """
    搜索 Agent 的长期记忆。

    当用户问"之前我说的那个..."时，Agent 通过此工具检索相关记忆。
    记忆系统使用混合检索（关键词 + 向量），返回最相关的记忆条目。

    参数:
        query: 搜索查询，如 "用户喜欢什么编程语言"
    """

    name = "memory_search"
    description = (
        "Search the agent's long-term memory for relevant information "
        "from previous conversations. Use this when the user refers to "
        "something from the past, like 'remember when I said...' or "
        "'what was that setting we changed last week?'."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query to find relevant memories",
            "required": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of memories to return. Defaults to memory config max_results.",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """检索相关长期记忆。"""
        query = params.get("query", "")
        if not query:
            return "Error: No search query provided"

        limit = int(params.get("limit", 0) or 0)
        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        results = memory_manager.search(query, limit=limit)
        if not results:
            return f"No memories found for: {query}"

        lines = [f"Found {len(results)} relevant memories:"]
        for i, result in enumerate(results, 1):
            lines.append(
                f"\n{i}. [{result.source}] {result.path}:{result.start_line}-{result.end_line} "
                f"(score: {result.score:.2f})\n{result.snippet}"
            )
        return "\n".join(lines)


class MemoryGetTool(BaseTool):
    """
    按 ID 或最近时间获取记忆条目。

    用于 Agent 在对话开始时加载最近几条记忆，保持上下文连贯性。

    参数:
        limit: 返回最近多少条记忆（默认 5）
    """

    name = "memory_get"
    description = (
        "Retrieve recent memories from the agent's long-term memory. "
        "Use this to get up to speed on what was discussed previously."
    )
    parameters = {
        "limit": {
            "type": "integer",
            "description": "Number of recent memories to retrieve (default 5, max 20)",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """获取最近的长期记忆条目。"""
        limit = min(max(int(params.get("limit", 5)), 1), 20)
        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        memories = memory_manager.get_recent(limit)
        if not memories:
            return "No memories stored yet."

        lines = [f"Most recent {len(memories)} memories:"]
        for i, memory in enumerate(memories, 1):
            lines.append(f"\n{i}. {memory[:300]}")
        return "\n".join(lines)


class MemoryRebuildTool(BaseTool):
    """
    重建长期记忆的 ChromaDB 向量索引。

    当 embedding 模型切换后，旧向量空间不再兼容，需要用 SQLite 中保存的
    文本块重新生成向量并写入 ChromaDB。
    """

    name = "memory_rebuild_vectors"
    description = (
        "Rebuild the ChromaDB vector index from SQLite memory chunks. "
        "Use this after changing the embedding model or provider."
    )
    parameters = {
        "confirm": {
            "type": "boolean",
            "description": "Must be true to confirm rebuilding the vector index.",
            "required": True,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """重建长期记忆的 ChromaDB 向量索引。"""
        if params.get("confirm") is not True:
            return (
                "Rebuild cancelled. Pass confirm=true to rebuild the "
                "ChromaDB vector index from SQLite memory chunks."
            )

        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        rebuilt = memory_manager.rebuild_vectors()
        return f"Rebuilt ChromaDB vector index with {rebuilt} memory chunks."


class MemoryStatusTool(BaseTool):
    """
    查看长期记忆系统当前状态。
    """

    name = "memory_status"
    description = (
        "Show the current status of the long-term memory system, including "
        "chunk count, indexed files, workspace, ChromaDB availability, and embedding status."
    )
    parameters = {}

    def execute(self, params: Dict[str, Any]) -> str:
        """查看长期记忆系统当前状态。"""
        agent = self._get_agent()
        memory_manager = getattr(agent, "memory_manager", None) if agent else None
        if not memory_manager:
            return (
                "[Memory system not available]\n"
                "Please initialize Agent with a memory_manager before using this tool."
            )

        status = memory_manager.get_status()
        return (
            "Memory system status:\n"
            f"- chunks: {status.get('chunks', 0)}\n"
            f"- files: {status.get('files', 0)}\n"
            f"- workspace: {status.get('workspace', '')}\n"
            f"- ChromaDB available: {status.get('chroma_available', False)}\n"
            f"- embedding available: {status.get('embedding_available', False)}\n"
            f"- embedding model: {status.get('embedding_model', '')}"
        )
