import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from agent.memory.summarizer import MemoryFlushManager


class MemoryFlushManagerRefineTest(unittest.TestCase):
    def test_refine_retries_same_daily_content_after_llm_failure(self):
        workspace = Path(tempfile.mkdtemp(prefix='myagent_refine_retry_test_'))
        memory_dir = workspace / 'memory'
        memory_dir.mkdir(parents=True, exist_ok=True)
        today_file = memory_dir / f'{datetime.now():%Y-%m-%d}.md'
        today_file.write_text('daily memory', encoding='utf-8')
        manager = MemoryFlushManager(workspace, llm_model=object())

        try:
            with patch.object(
                manager,
                '_call_llm',
                side_effect=[RuntimeError('temporary failure'), '- remembered'],
            ) as call_llm:
                self.assertFalse(manager.refine())
                self.assertTrue(manager.refine())

            self.assertEqual(call_llm.call_count, 2)
            self.assertEqual(
                (workspace / 'MEMORY.md').read_text(encoding='utf-8'),
                '- remembered\n',
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
