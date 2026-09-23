import unittest
from unittest.mock import MagicMock, patch

from agent.agent import Agent


class AgentContextTest(unittest.TestCase):
    def test_run_triggers_auto_refine_only_after_success(self):
        class FinalAnswerModel:
            def chat(self, messages, tools=None, stream=False):
                return {'text': 'done', 'tool_calls': [], 'usage': {}}

        memory_manager = MagicMock()
        memory_manager.search.return_value = []
        memory_manager.get_recent.return_value = []
        agent = Agent(model=FinalAnswerModel(), memory_manager=memory_manager)

        with patch('config.conf', return_value={'memory_auto_refine_enabled': True}):
            result = self._run_to_result(agent.run('hello'))

        self.assertTrue(result.success)
        memory_manager._run_auto_refine.assert_called_once_with(True)

        failing_memory_manager = MagicMock()
        failing_memory_manager.search.return_value = []
        failing_memory_manager.get_recent.return_value = []
        failing_model = MagicMock()
        failing_model.chat.side_effect = RuntimeError('model failed')
        failing_agent = Agent(
            model=failing_model,
            memory_manager=failing_memory_manager,
        )

        with patch('config.conf', return_value={'memory_auto_refine_enabled': True}):
            failed_result = self._run_to_result(failing_agent.run('hello'))

        self.assertFalse(failed_result.success)
        failing_memory_manager._run_auto_refine.assert_not_called()

    def _run_to_result(self, value):
        if not hasattr(value, "__next__"):
            return value
        try:
            next(value)
        except StopIteration as stop:
            return stop.value
        self.fail("Agent.run() generator yielded unexpectedly")

    def test_overflow_flush_discards_old_messages_when_token_limit_exceeded(self):
        memory_manager = MagicMock()
        agent = Agent(
            model=object(),
            memory_manager=memory_manager,
        )
        agent.max_context_tokens = 30
        system_message = {"role": "system", "content": "system"}
        old_messages = [
            {"role": "user", "content": "old user " * 40},
            {"role": "assistant", "content": "old assistant " * 40},
        ]
        recent_messages = [
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        agent.messages = [system_message] + old_messages + recent_messages

        agent._overflow_flush()

        memory_manager.flush.assert_called_once_with(
            old_messages,
            reason="overflow",
        )
        self.assertEqual(agent.messages, [system_message] + recent_messages)

    def test_threshold_flush_saves_old_messages_without_trimming_context(self):
        memory_manager = MagicMock()
        agent = Agent(
            model=object(),
            memory_manager=memory_manager,
        )
        agent.max_context_tokens = 500
        system_message = {"role": "system", "content": "system"}
        flushed_messages = [
            {"role": "user", "content": "old user " * 80},
        ]
        kept_old_messages = [
            {"role": "assistant", "content": "old assistant " * 80},
        ]
        recent_messages = [
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        original_messages = (
            [system_message]
            + flushed_messages
            + kept_old_messages
            + recent_messages
        )
        agent.messages = list(original_messages)

        agent._threshold_flush()

        memory_manager.flush.assert_called_once_with(
            flushed_messages,
            reason="threshold",
        )
        self.assertEqual(agent.messages, original_messages)
        self.assertEqual(
            agent._threshold_flushed_message_count,
            len(original_messages),
        )

    def test_threshold_flush_skips_when_token_limit_is_already_exceeded(self):
        memory_manager = MagicMock()
        agent = Agent(
            model=object(),
            memory_manager=memory_manager,
        )
        agent.max_context_tokens = 30
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old user " * 80},
            {"role": "assistant", "content": "old assistant " * 80},
            {"role": "user", "content": "recent question"},
        ]

        agent._threshold_flush()

        memory_manager.flush.assert_not_called()

    def test_run_calls_overflow_flush_before_model_call(self):
        class FinalAnswerModel:
            def chat(self, messages, tools=None, stream=False):
                return {
                    "text": "done",
                    "tool_calls": [],
                    "usage": {},
                }

        agent = Agent(model=FinalAnswerModel())
        agent._overflow_flush = MagicMock()

        self._run_to_result(agent.run("hello"))

        agent._overflow_flush.assert_called_once_with()

    def test_run_calls_threshold_flush_before_overflow_flush(self):
        class FinalAnswerModel:
            def chat(self, messages, tools=None, stream=False):
                return {
                    "text": "done",
                    "tool_calls": [],
                    "usage": {},
                }

        calls = []
        agent = Agent(model=FinalAnswerModel())
        agent._threshold_flush = MagicMock(side_effect=lambda: calls.append("threshold"))
        agent._overflow_flush = MagicMock(side_effect=lambda: calls.append("overflow"))

        self._run_to_result(agent.run("hello"))

        self.assertEqual(calls, ["threshold", "overflow"])


if __name__ == "__main__":
    unittest.main()
