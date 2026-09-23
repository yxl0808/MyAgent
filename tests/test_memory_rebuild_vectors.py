import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.memory.config import MemoryConfig
from agent.memory.manager import MemoryManager


class MemoryManagerRebuildVectorsTest(unittest.TestCase):
    def test_rebuild_vectors_recreates_chroma_collection_from_sqlite_chunks(self):
        with tempfile.TemporaryDirectory(prefix="myagent_rebuild_vectors_test_") as tmp:
            workspace = Path(tmp)
            config = MemoryConfig(
                workspace_dir=str(workspace),
                chroma_persist_dir=str(workspace / "chroma"),
                db_path=str(workspace / "memory" / "long-term" / "index.db"),
                auto_sync=False,
            )
            embedding_func = MagicMock(
                return_value=[
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            )
            with patch.object(MemoryManager, "_init_chroma", return_value=None):
                manager = MemoryManager(config=config, embedding_func=embedding_func)
            collection = MagicMock()
            chroma_client = MagicMock()
            chroma_client.get_or_create_collection.return_value = collection
            manager._chroma_client = chroma_client

            try:
                manager.storage.conn.executemany(
                    """
                    INSERT INTO chunks (
                        id, path, start_line, end_line, text, hash, source, importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            "chunk-1",
                            "MEMORY.md",
                            1,
                            2,
                            "用户喜欢 Python。",
                            "hash-1",
                            "memory",
                            0.8,
                        ),
                        (
                            "chunk-2",
                            "memory/2026-07-07.md",
                            3,
                            4,
                            "项目默认 embedding 是 bge-m3。",
                            "hash-2",
                            "memory",
                            0.6,
                        ),
                    ],
                )
                manager.storage.conn.commit()

                rebuilt = manager.rebuild_vectors()
            finally:
                manager.close()

            self.assertEqual(rebuilt, 2)
            chroma_client.delete_collection.assert_called_once_with("myagent_memory")
            chroma_client.get_or_create_collection.assert_called_once_with(
                name="myagent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            embedding_func.assert_called_once_with(
                ["用户喜欢 Python。", "项目默认 embedding 是 bge-m3。"]
            )
            collection.add.assert_called_once_with(
                ids=["chunk-1", "chunk-2"],
                documents=["用户喜欢 Python。", "项目默认 embedding 是 bge-m3。"],
                embeddings=[
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                metadatas=[
                    {
                        "path": "MEMORY.md",
                        "start_line": 1,
                        "end_line": 2,
                        "source": "memory",
                        "importance": 0.8,
                    },
                    {
                        "path": "memory/2026-07-07.md",
                        "start_line": 3,
                        "end_line": 4,
                        "source": "memory",
                        "importance": 0.6,
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
