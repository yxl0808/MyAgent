import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from agent.memory.config import MemoryConfig
from agent.memory.manager import MemoryManager


class MemoryManagerStatusTest(unittest.TestCase):
    def test_init_runs_sync_when_auto_sync_is_enabled(self):
        """验证 auto_sync 开启时初始化会执行一次同步。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_auto_sync_enabled_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=True,
        )
        manager = None
        try:
            with (
                patch.object(MemoryManager, "_init_chroma", return_value=None),
                patch.object(MemoryManager, "sync") as sync,
            ):
                manager = MemoryManager(config=config)

            sync.assert_called_once_with()
        finally:
            if manager is not None:
                manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_init_skips_sync_when_auto_sync_is_disabled(self):
        """验证 auto_sync 关闭时初始化不会执行同步。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_auto_sync_disabled_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
        )
        manager = None
        try:
            with (
                patch.object(MemoryManager, "_init_chroma", return_value=None),
                patch.object(MemoryManager, "sync") as sync,
            ):
                manager = MemoryManager(config=config)

            sync.assert_not_called()
        finally:
            if manager is not None:
                manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_init_continues_when_auto_sync_fails(self):
        """验证启动同步异常只记录警告，不阻断 MemoryManager 初始化。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_auto_sync_failure_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=True,
        )
        manager = None
        try:
            with (
                patch.object(MemoryManager, "_init_chroma", return_value=None),
                patch.object(
                    MemoryManager,
                    "sync",
                    side_effect=RuntimeError("sync failed"),
                ),
                self.assertLogs("agent.memory.manager", level="WARNING") as logs,
            ):
                manager = MemoryManager(config=config)

            self.assertIsInstance(manager, MemoryManager)
            self.assertTrue(
                any("Initial sync failed: sync failed" in line for line in logs.output)
            )
        finally:
            if manager is not None:
                manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_search_runs_sync_when_sync_on_search_is_enabled(self):
        """验证 sync_on_search 开启时搜索前会同步记忆文件。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_sync_on_search_enabled_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
            sync_on_search=True,
        )
        with patch.object(MemoryManager, "_init_chroma", return_value=None):
            manager = MemoryManager(config=config)

        try:
            with patch.object(manager, "sync") as sync:
                manager.search("query")

            sync.assert_called_once_with()
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_search_skips_sync_when_sync_on_search_is_disabled(self):
        """验证 sync_on_search 关闭时搜索前不会同步记忆文件。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_sync_on_search_disabled_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
            sync_on_search=False,
        )
        with patch.object(MemoryManager, "_init_chroma", return_value=None):
            manager = MemoryManager(config=config)

        try:
            with patch.object(manager, "sync") as sync:
                manager.search("query")

            sync.assert_not_called()
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_search_continues_when_pre_search_sync_fails(self):
        """验证搜索前同步异常只记录警告，不阻断搜索。"""
        workspace = Path(tempfile.mkdtemp(prefix="myagent_sync_on_search_failure_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
            sync_on_search=True,
        )
        with patch.object(MemoryManager, "_init_chroma", return_value=None):
            manager = MemoryManager(config=config)

        try:
            with (
                patch.object(manager, "sync", side_effect=RuntimeError("sync failed")),
                self.assertLogs("agent.memory.manager", level="WARNING") as logs,
            ):
                results = manager.search("query")

            self.assertEqual(results, [])
            self.assertTrue(
                any("Pre-search sync failed: sync failed" in line for line in logs.output)
            )
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_run_auto_refine_saves_state_only_after_success(self):
        workspace = Path(tempfile.mkdtemp(prefix='myagent_auto_refine_run_test_'))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / 'chroma'),
            db_path=str(workspace / 'memory' / 'long-term' / 'index.db'),
            auto_sync=False,
        )
        with patch.object(MemoryManager, '_init_chroma', return_value=None):
            manager = MemoryManager(config=config)

        now = datetime(2026, 7, 23, 18, 0)
        try:
            with (
                patch.object(manager, '_should_auto_refine', return_value=False),
                patch.object(manager, 'refine') as refine,
            ):
                self.assertFalse(manager._run_auto_refine(True, now=now))
                refine.assert_not_called()

            with (
                patch.object(manager, '_should_auto_refine', return_value=True),
                patch.object(manager, '_compute_daily_memory_hash', return_value='new'),
                patch.object(manager, 'refine', return_value=False),
                patch.object(manager, '_save_auto_refine_state') as save_state,
            ):
                self.assertFalse(manager._run_auto_refine(True, now=now))
                save_state.assert_not_called()

            with (
                patch.object(manager, '_should_auto_refine', return_value=True),
                patch.object(manager, '_compute_daily_memory_hash', return_value='new'),
                patch.object(manager, 'refine', return_value=True),
                patch.object(manager, '_load_auto_refine_state', return_value={'other': 1}),
                patch.object(manager, '_save_auto_refine_state', return_value=True) as save_state,
            ):
                self.assertTrue(manager._run_auto_refine(True, now=now))
                save_state.assert_called_once_with({
                    'other': 1,
                    'last_auto_success_date': '2026-07-23',
                    'last_processed_daily_hash': 'new',
                })
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_should_auto_refine_checks_all_conditions(self):
        workspace = Path(tempfile.mkdtemp(prefix='myagent_auto_refine_check_test_'))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / 'chroma'),
            db_path=str(workspace / 'memory' / 'long-term' / 'index.db'),
            auto_sync=False,
        )
        with patch.object(MemoryManager, '_init_chroma', return_value=None):
            manager = MemoryManager(config=config)

        cases = (
            ('disabled', False, datetime(2026, 7, 23, 20, 0), {}, 'new', False),
            ('before_cutoff', True, datetime(2026, 7, 23, 17, 59), {}, 'new', False),
            (
                'already_succeeded_today',
                True,
                datetime(2026, 7, 23, 18, 0),
                {'last_auto_success_date': '2026-07-23'},
                'new',
                False,
            ),
            ('no_daily_memory', True, datetime(2026, 7, 23, 18, 0), {}, '', False),
            (
                'same_daily_hash',
                True,
                datetime(2026, 7, 23, 18, 0),
                {'last_processed_daily_hash': 'same'},
                'same',
                False,
            ),
            (
                'ready',
                True,
                datetime(2026, 7, 23, 18, 0),
                {
                    'last_auto_success_date': '2026-07-22',
                    'last_processed_daily_hash': 'old',
                },
                'new',
                True,
            ),
        )

        try:
            for case_name, enabled, now, state, daily_hash, expected in cases:
                with self.subTest(case=case_name):
                    with (
                        patch.object(manager, '_load_auto_refine_state', return_value=state),
                        patch.object(manager, '_compute_daily_memory_hash', return_value=daily_hash),
                    ):
                        self.assertEqual(
                            manager._should_auto_refine(enabled, now=now),
                            expected,
                        )
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_compute_daily_memory_hash_tracks_only_daily_files_in_lookback(self):
        workspace = Path(tempfile.mkdtemp(prefix='myagent_daily_hash_test_'))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / 'chroma'),
            db_path=str(workspace / 'memory' / 'long-term' / 'index.db'),
            auto_sync=False,
        )
        with patch.object(MemoryManager, '_init_chroma', return_value=None):
            manager = MemoryManager(config=config)

        try:
            self.assertEqual(manager._compute_daily_memory_hash(), '')

            memory_dir = workspace / 'memory'
            memory_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now().date()
            today_file = memory_dir / f'{today:%Y-%m-%d}.md'
            yesterday_file = memory_dir / f'{today - timedelta(days=1):%Y-%m-%d}.md'
            outside_file = memory_dir / f'{today - timedelta(days=7):%Y-%m-%d}.md'

            today_file.write_text('today memory', encoding='utf-8')
            yesterday_file.write_text('yesterday memory', encoding='utf-8')
            outside_file.write_text('outside lookback', encoding='utf-8')
            (workspace / 'MEMORY.md').write_text('main memory', encoding='utf-8')

            daily_hash = manager._compute_daily_memory_hash(lookback_days=7)
            self.assertEqual(len(daily_hash), 64)
            self.assertEqual(
                manager._compute_daily_memory_hash(lookback_days=7),
                daily_hash,
            )

            outside_file.write_text('changed outside', encoding='utf-8')
            (workspace / 'MEMORY.md').write_text('changed main', encoding='utf-8')
            self.assertEqual(
                manager._compute_daily_memory_hash(lookback_days=7),
                daily_hash,
            )

            today_file.write_text('changed today', encoding='utf-8')
            self.assertNotEqual(
                manager._compute_daily_memory_hash(lookback_days=7),
                daily_hash,
            )
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_save_auto_refine_state_writes_atomically_and_handles_invalid_state(self):
        workspace = Path(tempfile.mkdtemp(prefix='myagent_auto_refine_save_test_'))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / 'chroma'),
            db_path=str(workspace / 'memory' / 'long-term' / 'index.db'),
            auto_sync=False,
        )
        with patch.object(MemoryManager, '_init_chroma', return_value=None):
            manager = MemoryManager(config=config)

        try:
            state = {
                'last_auto_success_date': '2026-07-10',
                'last_processed_daily_hash': 'abc123',
            }
            state_file = manager._get_auto_refine_state_file()
            temp_file = state_file.with_name(f'{state_file.name}.tmp')

            self.assertTrue(manager._save_auto_refine_state(state))
            self.assertEqual(
                json.loads(state_file.read_text(encoding='utf-8')),
                state,
            )
            self.assertFalse(temp_file.exists())
            self.assertFalse(manager._save_auto_refine_state({'invalid': object()}))
            self.assertFalse(temp_file.exists())
            self.assertEqual(manager._load_auto_refine_state(), state)
        finally:
            manager.close()
            shutil.rmtree(workspace, ignore_errors=True)

    def test_load_auto_refine_state_handles_valid_and_unusable_files(self):
        cases = {
            'missing': (None, {}),
            'empty': ('', {}),
            'invalid': ('{not-json', {}),
            'array': ('[]', {}),
            'null': ('null', {}),
            'valid': (
                json.dumps({
                    'last_auto_success_date': '2026-07-10',
                    'last_processed_daily_hash': 'abc123',
                }),
                {
                    'last_auto_success_date': '2026-07-10',
                    'last_processed_daily_hash': 'abc123',
                },
            ),
        }

        for case_name, (contents, expected) in cases.items():
            with self.subTest(case=case_name):
                workspace = Path(tempfile.mkdtemp(prefix='myagent_auto_refine_load_test_'))
                config = MemoryConfig(
                    workspace_dir=str(workspace),
                    chroma_persist_dir=str(workspace / 'chroma'),
                    db_path=str(workspace / 'memory' / 'long-term' / 'index.db'),
                    auto_sync=False,
                )
                with patch.object(MemoryManager, '_init_chroma', return_value=None):
                    manager = MemoryManager(config=config)

                try:
                    state_file = manager._get_auto_refine_state_file()
                    if contents is not None:
                        state_file.parent.mkdir(parents=True, exist_ok=True)
                        state_file.write_text(contents, encoding='utf-8')
                    self.assertEqual(manager._load_auto_refine_state(), expected)
                finally:
                    manager.close()
                    shutil.rmtree(workspace, ignore_errors=True)

    def test_get_auto_refine_state_file_uses_workspace_memory_dir(self):
        workspace = Path(tempfile.mkdtemp(prefix="myagent_auto_refine_state_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
        )

        with patch.object(MemoryManager, "_init_chroma", return_value=None):
            manager = MemoryManager(config=config)

        try:
            state_file = manager._get_auto_refine_state_file()
        finally:
            manager.close()

        self.assertEqual(
            state_file,
            workspace.resolve() / "memory" / ".auto-refine-state.json",
        )

    def test_get_status_marks_embedding_unavailable_when_probe_fails(self):
        workspace = Path(tempfile.mkdtemp(prefix="myagent_status_test_"))
        config = MemoryConfig(
            workspace_dir=str(workspace),
            chroma_persist_dir=str(workspace / "chroma"),
            db_path=str(workspace / "memory" / "long-term" / "index.db"),
            auto_sync=False,
        )

        def broken_embedding(_texts):
            raise RuntimeError("embedding service unavailable")

        manager = MemoryManager(config=config, embedding_func=broken_embedding)
        try:
            status = manager.get_status()
        finally:
            manager.close()

        self.assertFalse(status["embedding_available"])


if __name__ == "__main__":
    unittest.main()
