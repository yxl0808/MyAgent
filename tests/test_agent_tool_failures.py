import unittest

from agent.agent import Agent
from agent.tools.base import BaseTool


class SequenceModel:
    """按顺序返回非流式模型响应的测试模型。"""

    def __init__(self, responses):
        """保存预设响应和调用次数。"""
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None, stream=False):
        """返回下一条预设响应。"""
        self.calls += 1
        return self.responses.pop(0)


class ScriptedTool(BaseTool):
    """返回预设成功或失败文本的测试工具。"""

    name = "scripted"
    description = "Return scripted output"
    parameters = {}

    def __init__(self, outputs):
        """保存每次调用应返回的文本。"""
        self.outputs = list(outputs)
        self.calls = 0

    def execute(self, params):
        """返回下一条预设工具输出。"""
        self.calls += 1
        return self.outputs.pop(0)


def tool_response(call_id, arguments):
    """构造单个工具调用的非流式模型响应。"""
    return {
        "text": "",
        "tool_calls": [{
            "id": call_id,
            "name": "scripted",
            "arguments": arguments,
        }],
        "usage": {},
    }


def run_to_result(value):
    """提取非流式 Agent.run() 生成器的最终结果。"""
    try:
        next(value)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("Expected a non-stream result generator")


class AgentToolFailureTest(unittest.TestCase):
    def test_stops_after_three_consecutive_failures_from_same_tool(self):
        """验证同一工具连续三次失败会在下一次模型调用前中止。"""
        model = SequenceModel([
            tool_response("call_1", {"attempt": 1}),
            tool_response("call_2", {"attempt": 2}),
            tool_response("call_3", {"attempt": 3}),
            {"text": "must not be requested", "tool_calls": [], "usage": {}},
        ])
        tool = ScriptedTool([
            "[exit code 1]\nfirst failure",
            "[exit code 1]\nsecond failure",
            "[exit code 1]\nthird failure",
        ])

        result = run_to_result(Agent(model=model, tools=[tool]).run("run task"))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "tool_failure_limit")
        self.assertEqual(result.step_count, 3)
        self.assertEqual(model.calls, 3)
        self.assertEqual(tool.calls, 3)
        self.assertIn("连续失败 3 次", result.final_answer)
        self.assertTrue(all(not action.tool_result.success for action in result.actions))

    def test_successful_call_resets_consecutive_failure_counter(self):
        """验证一次成功调用会清除同一工具已有的连续失败计数。"""
        model = SequenceModel([
            tool_response("call_1", {"attempt": 1}),
            tool_response("call_2", {"attempt": 2}),
            tool_response("call_3", {"attempt": 3}),
            tool_response("call_4", {"attempt": 4}),
            {"text": "completed after recovery", "tool_calls": [], "usage": {}},
        ])
        tool = ScriptedTool([
            "Error: first failure",
            "recovered",
            "Error: second failure",
            "Error: third failure",
        ])

        result = run_to_result(Agent(model=model, tools=[tool]).run("run task"))

        self.assertTrue(result.success)
        self.assertEqual(result.final_answer, "completed after recovery")
        self.assertEqual(model.calls, 5)
        self.assertEqual(tool.calls, 4)


if __name__ == "__main__":
    unittest.main()
