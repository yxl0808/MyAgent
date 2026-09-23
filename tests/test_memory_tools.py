import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.memory.storage import SearchResult
from agent.tools.memory import MemoryFlushTool
from agent.tools.memory import MemoryGetTool
from agent.tools.memory import MemoryRebuildTool
from agent.tools.memory import MemoryRefineTool
from agent.tools.memory import MemorySearchTool
from agent.tools.memory import MemoryStatusTool


class MemoryToolsTest(unittest.TestCase):
    def test_memory_flush_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.flush.return_value = True
        messages = [
            {"role": "user", "content": "remember this"},
            {"role": "assistant", "content": "stored"},
        ]
        tool = MemoryFlushTool()
        tool._agent = SimpleNamespace(
            memory_manager=memory_manager,
            messages=messages,
        )

        result = tool.execute({"reason": "manual"})

        memory_manager.flush.assert_called_once_with(messages, reason="manual")
        self.assertEqual(
            result,
            "Flushed 2 conversation messages into daily memory.",
        )

    def test_memory_refine_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.refine.return_value = True
        tool = MemoryRefineTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({"lookback_days": 3, "force": True})

        memory_manager.refine.assert_called_once_with(
            lookback_days=3,
            force=True,
        )
        self.assertEqual(
            result,
            "Refined daily memory into MEMORY.md.",
        )

    def test_memory_rebuild_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.rebuild_vectors.return_value = 3
        tool = MemoryRebuildTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({"confirm": True})

        memory_manager.rebuild_vectors.assert_called_once_with()
        self.assertEqual(
            result,
            "Rebuilt ChromaDB vector index with 3 memory chunks.",
        )

    def test_memory_rebuild_tool_requires_confirmation(self):
        memory_manager = MagicMock()
        tool = MemoryRebuildTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({"confirm": False})

        memory_manager.rebuild_vectors.assert_not_called()
        self.assertEqual(
            result,
            "Rebuild cancelled. Pass confirm=true to rebuild the "
            "ChromaDB vector index from SQLite memory chunks.",
        )

    def test_memory_search_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.search.return_value = [
            SearchResult(
                path="MEMORY.md",
                start_line=1,
                end_line=2,
                score=0.87,
                snippet="用户喜欢 Python。",
                source="memory",
            )
        ]
        tool = MemorySearchTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({"query": "用户喜欢什么语言", "limit": 3})

        memory_manager.search.assert_called_once_with("用户喜欢什么语言", limit=3)
        self.assertIn("Found 1 relevant memories:", result)
        self.assertIn("[memory] MEMORY.md:1-2 (score: 0.87)", result)
        self.assertIn("用户喜欢 Python。", result)

    def test_memory_get_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.get_recent.return_value = [
            "第一条长期记忆",
            "第二条长期记忆",
        ]
        tool = MemoryGetTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({"limit": 2})

        memory_manager.get_recent.assert_called_once_with(2)
        self.assertIn("Most recent 2 memories:", result)
        self.assertIn("1. 第一条长期记忆", result)
        self.assertIn("2. 第二条长期记忆", result)

    def test_memory_status_tool_calls_agent_memory_manager(self):
        memory_manager = MagicMock()
        memory_manager.get_status.return_value = {
            "chunks": 7,
            "files": 2,
            "workspace": "E:/item/CowAgent-master/MyAgent",
            "chroma_available": True,
            "embedding_available": True,
            "embedding_model": "bge-m3",
        }
        tool = MemoryStatusTool()
        tool._agent = SimpleNamespace(memory_manager=memory_manager)

        result = tool.execute({})

        memory_manager.get_status.assert_called_once_with()
        self.assertIn("Memory system status:", result)
        self.assertIn("- chunks: 7", result)
        self.assertIn("- files: 2", result)
        self.assertIn("- workspace: E:/item/CowAgent-master/MyAgent", result)
        self.assertIn("- ChromaDB available: True", result)
        self.assertIn("- embedding available: True", result)
        self.assertIn("- embedding model: bge-m3", result)


if __name__ == "__main__":
    unittest.main()
