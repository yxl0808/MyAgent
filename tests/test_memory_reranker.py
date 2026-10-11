import math
import tempfile
import unittest
from unittest.mock import patch

from agent.memory.config import MemoryConfig
from agent.memory.manager import MemoryManager
from agent.memory.reranker import LocalReranker


class MemoryRerankerTest(unittest.TestCase):
    def test_local_adapter_batches_pairs_and_loads_once(self):
        from unittest.mock import MagicMock
        import sys
        backend = MagicMock()
        backend.predict.return_value = [0.2, 0.9]
        module = MagicMock()
        module.CrossEncoder.return_value = backend
        with patch.dict(sys.modules, {'sentence_transformers': module}):
            scorer = LocalReranker(batch_size=2)
            self.assertEqual(scorer('q', ['a','b']), [0.2,0.9])
            scorer('q2', ['a','b'])
            module.CrossEncoder.assert_called_once()
            backend.predict.assert_any_call([('q','a'),('q','b')], batch_size=2,
                                            show_progress_bar=False)

    def test_factory_passes_reranker_configuration(self):
        from agent.memory.factory import create_memory_manager
        with tempfile.TemporaryDirectory() as root, patch('config.conf', return_value={'memory_reranker_enabled': True,
                   'memory_reranker_candidates': 30}), \
             patch('agent.memory.factory.MemoryManager') as manager, \
             patch('agent.memory.factory.get_embedding_func'):
            create_memory_manager(workspace_dir=root)
            cfg = manager.call_args.kwargs['config']
            self.assertTrue(cfg.reranker_enabled)
            self.assertEqual(cfg.reranker_candidates,30)

    def make_manager(self, root, scorer):
        cfg = MemoryConfig(workspace_dir=root, auto_sync=False, sync_on_search=False,
                           reranker_enabled=True, reranker_candidates=20)
        with patch.object(MemoryManager, '_init_chroma'):
            manager = MemoryManager(config=cfg, reranker=scorer)
        self.addCleanup(manager.close)
        manager.storage.save_chunks_batch([
            dict(id='a', path='a.md', start_line=1, end_line=2,
                 text='keyword ' + 'x' * 600 + ' full evidence', hash='a'),
            dict(id='b', path='b.md', start_line=1, end_line=2,
                 text='keyword relevant', hash='b'),
        ])
        return manager

    def test_scores_full_text_before_truncating_to_limit(self):
        seen = []
        def score(query, texts):
            seen.extend(texts)
            return [0.1 if 'full evidence' in t else 0.9 for t in texts]
        with tempfile.TemporaryDirectory() as root:
            manager = self.make_manager(root, score)
            result = manager.search('keyword', limit=1)
            self.assertEqual(result[0].path, 'b.md')
            self.assertTrue(any('full evidence' in t for t in seen))
            self.assertEqual(result[0].score, 0.9)
            manager.close()

    def test_invalid_or_failed_scoring_preserves_baseline(self):
        def fail(*args):
            raise RuntimeError('offline')
        for scorer in (fail, lambda q,t: [math.nan]*len(t), lambda q,t: [0.5]):
            with self.subTest(scorer=scorer), tempfile.TemporaryDirectory() as root:
                manager = self.make_manager(root, scorer)
                manager.config.reranker_enabled = False
                baseline = manager.search('keyword', limit=2)
                manager.config.reranker_enabled = True
                self.assertEqual(manager.search('keyword', limit=2), baseline)
                manager.close()

    def test_disabled_reranker_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as root:
            def fail(*args):
                self.fail('scorer must not run')
            manager = self.make_manager(root, fail)
            manager.config.reranker_enabled = False
            self.assertEqual(len(manager.search('keyword', limit=2)), 2)
            manager.close()


if __name__ == '__main__':
    unittest.main()
