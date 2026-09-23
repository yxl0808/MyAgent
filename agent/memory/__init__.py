# encoding:utf-8
"""
MyAgent 记忆系统 — 混合检索 + 对话摘要 + 记忆蒸馏。
"""

from agent.memory.config import MemoryConfig, get_embedding_func
from agent.memory.factory import create_memory_manager
from agent.memory.manager import MemoryManager
from agent.memory.storage import SearchResult

__all__ = [
    "create_memory_manager",
    "MemoryConfig",
    "MemoryManager",
    "SearchResult",
    "get_embedding_func",
]
