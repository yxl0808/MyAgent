import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from agent.agent import Agent
from agent.skills.manager import SkillManager
from agent.tools.base import BaseTool


class CapturingModel:
    """记录 Agent 发给假模型的消息，并按顺序返回预设响应。"""

    def __init__(self, responses=None):
        """初始化预设响应队列和调用记录。"""
        self.responses = list(
            responses
            or [{"text": "done", "tool_calls": [], "usage": {}}]
        )
        self.calls = []

    def chat(self, messages, tools=None, stream=False):
        """保存一次模型调用输入并返回下一条预设响应。"""
        self.calls.append(deepcopy(messages))
        return self.responses.pop(0)


class EchoTool(BaseTool):
    """提供确定性回显，用于验证多步 ReAct Skill 注入。"""

    name = "echo"
    description = "Echo text"
    parameters = {
        "text": {
            "type": "string",
            "description": "Text to echo",
            "required": True,
        }
    }

    def execute(self, params):
        """返回传入文本，作为无副作用的工具执行结果。"""
        return params["text"]


def run_to_result(value):
    """解析非流式 Agent.run 生成器并取出最终 AgentResult。"""
    try:
        next(value)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("Non-stream Agent.run() yielded unexpectedly")


class AgentSkillCommandTest(unittest.TestCase):
    """验证显式 Skill 命令、单轮注入和直接返回行为。"""

    def _manager_with_skill(self, root, name="debug-skill"):
        """在临时工作区创建一个最小 SkillManager 测试夹具。"""
        builtin = Path(root) / "builtin"
        custom = Path(root) / "workspace" / "skills"
        skill_dir = custom / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            "description: Debug a failure\n"
            "---\n"
            "# Required workflow\n\nFind the root cause first.\n",
            encoding="utf-8",
        )
        return SkillManager(builtin, custom), skill_dir

    def test_plain_message_does_not_inject_or_refresh_skills(self):
        """验证普通消息既不刷新也不注入 Skill。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _ = self._manager_with_skill(temp_dir)
            model = CapturingModel()
            agent = Agent(model=model, skill_manager=manager)

            result = run_to_result(agent.run("please debug this"))

        self.assertTrue(result.success)
        system_prompt = model.calls[0][0]["content"]
        self.assertNotIn("## Active Skill", system_prompt)
        self.assertEqual(dict(manager.snapshot.skills), {})

    def test_explicit_skill_injects_body_root_and_clean_task(self):
        """验证显式命令注入正文、根目录和裁剪后的任务。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, skill_dir = self._manager_with_skill(temp_dir)
            model = CapturingModel()
            agent = Agent(model=model, skill_manager=manager)

            result = run_to_result(
                agent.run("/debug-skill\n  investigate test failure\ncarefully  ")
            )

        self.assertTrue(result.success)
        messages = model.calls[0]
        prompt = messages[0]["content"]
        self.assertEqual(
            messages[-1],
            {"role": "user", "content": "investigate test failure\ncarefully"},
        )
        self.assertEqual(prompt.count("## Active Skill"), 1)
        self.assertIn("Find the root cause first.", prompt)
        self.assertIn(str(skill_dir.resolve()), prompt)

    def test_skill_stays_for_tool_loop_then_disappears_next_turn(self):
        """验证工具循环保留 Skill，而下一轮普通消息会移除它。"""
        responses = [
            {
                "text": "checking",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "echo",
                        "arguments": {"text": "evidence"},
                    }
                ],
                "usage": {},
            },
            {"text": "fixed", "tool_calls": [], "usage": {}},
            {"text": "next", "tool_calls": [], "usage": {}},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _ = self._manager_with_skill(temp_dir)
            model = CapturingModel(responses)
            agent = Agent(
                model=model,
                tools=[EchoTool()],
                skill_manager=manager,
            )

            run_to_result(agent.run("/debug-skill investigate"))
            run_to_result(agent.run("ordinary follow-up"))

        self.assertEqual(model.calls[0][0]["content"].count("## Active Skill"), 1)
        self.assertEqual(model.calls[1][0]["content"].count("## Active Skill"), 1)
        self.assertNotIn("## Active Skill", model.calls[2][0]["content"])
        self.assertNotIn("Find the root cause first.", model.calls[2][0]["content"])

    def test_direct_commands_do_not_call_model_or_write_history(self):
        """验证列表和错误命令直接返回，不访问模型或会话历史。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _ = self._manager_with_skill(temp_dir)
            cases = (
                (manager, "/skills", True, "Available skills:"),
                (manager, "/skills extra", False, "Usage: /skills"),
                (manager, "/missing-skill task", False, "Unknown skill"),
                (manager, "/debug-skill   ", False, "requires a task"),
                (None, "/skills", False, "Skill functionality is disabled"),
            )
            for skill_manager, command, success, expected in cases:
                with self.subTest(command=command, manager=skill_manager):
                    model = CapturingModel()
                    agent = Agent(model=model, skill_manager=skill_manager)
                    result = run_to_result(agent.run(command))
                    self.assertEqual(result.success, success)
                    self.assertIn(expected, result.final_answer)
                    self.assertEqual(model.calls, [])
                    self.assertEqual(agent.messages, [])

    def test_stream_list_returns_one_done_event_without_model_call(self):
        """验证流式 /skills 只发送一个 done 事件且不调用模型。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager, _ = self._manager_with_skill(temp_dir)
            model = CapturingModel()
            agent = Agent(model=model, skill_manager=manager)

            events = list(agent.run("/skills", stream=True))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "done")
        self.assertTrue(events[0]["result"].success)
        self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
