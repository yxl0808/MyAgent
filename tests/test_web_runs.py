import copy
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from channel.web.run_manager import (
    AgentBusyError,
    RunNotFoundError,
    RunState,
    WebRunManager,
)
from channel.web.store import WebConsoleStore


class RecordingAgent:
    """生成确定性完成事件并记录每轮恢复上下文的测试 Agent。"""

    def __init__(self):
        """创建空的 Agent 消息和上下文记录列表。"""
        self.messages = []
        self.input_contexts = []

    def run(self, user_message, stream=False, cancel_event=None):
        """生成一条文本事件和一条成功完成事件。"""
        self.input_contexts.append(copy.deepcopy(self.messages))
        self.messages.append({"role": "user", "content": user_message})
        answer = f"回答：{user_message}"
        yield {"type": "text", "content": answer, "step": 1}
        self.messages.append({"role": "assistant", "content": answer})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer=answer, success=True),
        }


class SystemPromptAgent(RecordingAgent):
    """模拟真实 Agent 在历史首位插入系统提示词的行为。"""

    def run(self, user_message, stream=False, cancel_event=None):
        """插入系统消息后生成确定性成功回答。"""
        self.messages.insert(0, {"role": "system", "content": "系统提示"})
        yield from super().run(user_message, stream=stream, cancel_event=cancel_event)


class ToolAgent:
    """生成工具事件和完成事件的确定性测试 Agent。"""

    def __init__(self):
        """创建空的 Agent 消息列表。"""
        self.messages = []

    def run(self, user_message, stream=False, cancel_event=None):
        """生成工具开始、结束、文本和完成事件。"""
        self.messages.append({"role": "user", "content": user_message})
        yield {"type": "tool_start", "name": "lookup", "args": {"q": "test"}}
        yield {
            "type": "tool_end",
            "name": "lookup",
            "output": "found",
            "success": True,
        }
        yield {"type": "text", "content": "工具已完成", "step": 1}
        self.messages.append({"role": "assistant", "content": "工具已完成"})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer="工具已完成", success=True),
        }


class StoppedAgent:
    """生成部分文本后停止的确定性测试 Agent。"""

    def __init__(self):
        """创建空的 Agent 消息列表。"""
        self.messages = []

    def run(self, user_message, stream=False, cancel_event=None):
        """生成部分文本和停止事件。"""
        self.messages.append({"role": "user", "content": user_message})
        yield {"type": "text", "content": "部分回答", "step": 1}
        yield {"type": "stopped"}


class FailedAgent:
    """生成失败完成事件的确定性测试 Agent。"""

    def __init__(self):
        """创建空的 Agent 消息列表。"""
        self.messages = []

    def run(self, user_message, stream=False, cancel_event=None):
        """生成失败的完成事件。"""
        self.messages.append({"role": "user", "content": user_message})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer="模型失败", success=False),
        }


class BlockingAgent:
    """在外部释放前保持运行中的确定性测试 Agent。"""

    def __init__(self):
        """创建启动与释放同步事件。"""
        self.messages = []
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, user_message, stream=False, cancel_event=None):
        """等待释放后生成成功完成事件。"""
        self.messages.append({"role": "user", "content": user_message})
        self.started.set()
        self.release.wait(2.0)
        self.messages.append({"role": "assistant", "content": "完成"})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer="完成", success=True),
        }


class WebRunManagerTest(unittest.TestCase):
    def setUp(self):
        """为每项测试创建隔离的已初始化 Web Store。"""
        self.temp_dir = tempfile.TemporaryDirectory(prefix="myagent_web_runs_")
        self.store = WebConsoleStore(Path(self.temp_dir.name) / "web-console.db")
        self.store.initialize()

    def tearDown(self):
        """释放每项测试创建的临时 Web Store。"""
        self.temp_dir.cleanup()

    def _session(self):
        """创建一个用于运行测试的 Web 会话。"""
        return self.store.create_session()

    def _wait_for_run(self, state):
        """等待后台运行线程结束并确认其没有卡住。"""
        self.assertIsNotNone(state.worker)
        state.worker.join(2.0)
        self.assertFalse(state.worker.is_alive())

    def test_complete_run_derives_title_and_persists_context(self):
        """验证首次运行更新标题并持久化完整上下文。"""
        session = self._session()
        manager = WebRunManager(RecordingAgent(), self.store)

        state = manager.start_run(session.id, "  设计方案  ")
        self._wait_for_run(state)

        messages = self.store.list_messages(session.id)
        self.assertEqual(self.store.get_session(session.id).title, "设计方案")
        self.assertEqual([message.sequence for message in messages], [1, 2])
        self.assertEqual(messages[0].payload["content"], "设计方案")
        self.assertEqual(messages[1].payload["status"], "complete")
        self.assertIs(messages[1].payload["include_in_context"], True)
        self.assertEqual([event.type for event in state.events], ["text", "done"])
        self.assertFalse(manager.is_session_active(session.id))

    def test_context_reconstruction_stays_within_its_session(self):
        """验证后续运行只恢复所属会话的完整上下文。"""
        agent = RecordingAgent()
        manager = WebRunManager(agent, self.store)
        first_session = self._session()
        second_session = self._session()

        self._wait_for_run(manager.start_run(first_session.id, "第一会话"))
        self._wait_for_run(manager.start_run(second_session.id, "第二会话"))
        self._wait_for_run(manager.start_run(first_session.id, "继续第一会话"))

        restored = agent.input_contexts[2]
        self.assertEqual(
            restored,
            [
                {"role": "user", "content": "第一会话"},
                {"role": "assistant", "content": "回答：第一会话"},
            ],
        )

    def test_system_prompt_is_not_persisted_as_assistant_context(self):
        """验证真实 Agent 插入系统消息不会污染保存的回答上下文。"""
        session = self._session()
        manager = WebRunManager(SystemPromptAgent(), self.store)

        self._wait_for_run(manager.start_run(session.id, "首轮问题"))

        assistant = self.store.list_messages(session.id)[-1].payload
        self.assertEqual(
            assistant["context_messages"],
            [{"role": "assistant", "content": "回答：首轮问题"}],
        )

    def test_tool_events_are_retained_with_terminal_message(self):
        """验证工具事件会进入 SSE 缓冲和 Assistant 消息账本。"""
        session = self._session()
        manager = WebRunManager(ToolAgent(), self.store)

        state = manager.start_run(session.id, "执行工具")
        self._wait_for_run(state)

        messages = self.store.list_messages(session.id)
        assistant = messages[-1].payload
        self.assertEqual(
            [event.type for event in state.events],
            ["tool_start", "tool_end", "text", "done"],
        )
        self.assertEqual(
            [event["type"] for event in assistant["execution_events"]],
            ["tool_start", "tool_end"],
        )

    def test_stopped_and_failed_runs_exclude_context(self):
        """验证停止和失败消息可见但不会进入后续上下文。"""
        stopped_session = self._session()
        stopped_manager = WebRunManager(StoppedAgent(), self.store)
        self._wait_for_run(stopped_manager.start_run(stopped_session.id, "停止测试"))

        failed_session = self._session()
        failed_manager = WebRunManager(FailedAgent(), self.store)
        self._wait_for_run(failed_manager.start_run(failed_session.id, "失败测试"))

        stopped = self.store.list_messages(stopped_session.id)[-1].payload
        failed = self.store.list_messages(failed_session.id)[-1].payload
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(failed["status"], "error")
        self.assertIs(stopped["include_in_context"], False)
        self.assertIs(failed["include_in_context"], False)
        self.assertEqual(stopped["context_messages"], [])
        self.assertEqual(failed["context_messages"], [])

    def test_active_run_blocks_second_generation(self):
        """验证共享 Agent 运行期间拒绝新的生成请求。"""
        agent = BlockingAgent()
        manager = WebRunManager(agent, self.store)
        first_session = self._session()
        second_session = self._session()

        state = manager.start_run(first_session.id, "保持运行")
        self.assertTrue(agent.started.wait(1.0))
        with self.assertRaises(AgentBusyError):
            manager.start_run(second_session.id, "第二请求")
        agent.release.set()
        self._wait_for_run(state)

    def test_run_state_replay_bound_and_expiry(self):
        """验证事件重放上限与运行过期清理。"""
        state = RunState("run-1", "session-1", "消息", 0, max_events=2)
        state.publish("text", {"content": "一"})
        state.publish("text", {"content": "二"})
        state.publish("done", {"status": "complete"})
        self.assertEqual([event.id for event in state.events_after(0)], [2, 3])
        self.assertTrue(state.finished)

        manager = WebRunManager(RecordingAgent(), self.store, retention_seconds=0.0)
        manager._runs[state.run_id] = state
        with self.assertRaises(RunNotFoundError):
            manager.get_run(state.run_id)


if __name__ == "__main__":
    unittest.main()
