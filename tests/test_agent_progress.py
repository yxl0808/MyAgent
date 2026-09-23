import copy
import unittest

from agent.agent import Agent
from agent.tools.base import BaseTool


class ProgressSequenceModel:
    """按顺序返回非流式响应并保存请求快照的测试模型。"""

    def __init__(self, responses):
        """保存预设响应、调用次数和消息快照。"""
        self.responses = list(responses)
        self.calls = 0
        self.message_snapshots = []

    def chat(self, messages, tools=None, stream=False):
        """保存本次请求的消息并返回下一条响应。"""
        self.calls += 1
        self.message_snapshots.append(copy.deepcopy(messages))
        return self.responses.pop(0)


class StreamingProgressModel:
    """按顺序返回流式响应的测试模型。"""

    def __init__(self, factories):
        """保存流式响应工厂和调用次数。"""
        self.factories = list(factories)
        self.calls = 0

    def chat(self, messages, tools=None, stream=False):
        """返回下一条流式响应生成器。"""
        self.calls += 1
        return self.factories.pop(0)()


class ValidatingSequenceModel(ProgressSequenceModel):
    """验证每条 assistant tool_calls 都有完整工具响应的测试模型。"""

    def chat(self, messages, tools=None, stream=False):
        for index, message in enumerate(messages):
            tool_calls = message.get("tool_calls", [])
            if not tool_calls:
                continue
            following = messages[index + 1:index + 1 + len(tool_calls)]
            self.assert_complete_tool_messages(tool_calls, following)
        return super().chat(messages, tools=tools, stream=stream)

    @staticmethod
    def assert_complete_tool_messages(tool_calls, following):
        """验证每个工具调用 ID 都紧跟对应的 tool 消息。"""
        if len(following) != len(tool_calls):
            raise AssertionError("assistant tool_calls has insufficient tool messages")
        expected_ids = [call.get("id") for call in tool_calls]
        actual_ids = [message.get("tool_call_id") for message in following]
        if any(message.get("role") != "tool" for message in following):
            raise AssertionError("assistant tool_calls is not followed by tool messages")
        if expected_ids != actual_ids:
            raise AssertionError("tool_call_id responses do not match assistant tool_calls")


class ProgressTool(BaseTool):
    """返回固定结果的测试工具。"""

    name = "progress_tool"
    description = "Return a test result"
    parameters = {}

    def __init__(self, output):
        """保存工具输出。"""
        self.output = output

    def execute(self, params):
        """返回固定工具结果。"""
        return self.output


class CountingProgressTool(ProgressTool):
    """记录工具实际执行次数的测试工具。"""

    def __init__(self, output):
        super().__init__(output)
        self.calls = 0

    def execute(self, params):
        self.calls += 1
        return super().execute(params)


def tool_response(call_id, text, arguments=None, tool_name="progress_tool"):
    """构造一条带工具调用的模型响应。"""
    return {
        "text": text,
        "tool_calls": [{
            "id": call_id,
            "name": tool_name,
            "arguments": {} if arguments is None else arguments,
        }],
        "usage": {},
    }


def multi_tool_response(call_ids, text, arguments=None, tool_name="progress_tool"):
    """构造一条包含多个工具调用的模型响应。"""
    arguments = {} if arguments is None else arguments
    return {
        "text": text,
        "tool_calls": [
            {
                "id": call_id,
                "name": tool_name,
                "arguments": arguments,
            }
            for call_id in call_ids
        ],
        "usage": {},
    }


def run_to_result(value):
    """提取非流式 Agent.run() 生成器的最终结果。"""
    try:
        next(value)
    except StopIteration as stop:
        return stop.value
    raise AssertionError("Expected a non-stream result generator")


class AgentProgressTest(unittest.TestCase):
    def test_early_stop_completes_all_tool_call_responses(self):
        """验证提前停止时仍为同一 assistant 的全部调用补齐响应。"""
        model = ValidatingSequenceModel([
            tool_response("call_1", "第一次", {"query": "same"}),
            tool_response("call_2", "第二次", {"query": "same"}),
            multi_tool_response(
                ["call_3", "call_4"],
                "准备再次调用",
                {"query": "same"},
            ),
            {"text": "下一轮完成", "tool_calls": [], "usage": {}},
        ])
        tool = CountingProgressTool("结果")
        agent = Agent(model=model, tools=[tool])

        first_result = run_to_result(agent.run("repeat"))
        second_result = run_to_result(agent.run("continue"))

        self.assertFalse(first_result.success)
        self.assertEqual(first_result.error, "duplicate_tool_call")
        self.assertTrue(second_result.success)
        self.assertEqual(tool.calls, 2)
        self.assertEqual(model.calls, 4)

    def test_stops_before_third_identical_tool_call(self):
        """验证连续三次相同工具调用只实际执行两次。"""
        model = ProgressSequenceModel([
            tool_response("call_1", "第一次", {"query": "same"}),
            tool_response("call_2", "第二次", {"query": "same"}),
            tool_response("call_3", "第三次", {"query": "same"}),
            {"text": "不得继续调用", "tool_calls": [], "usage": {}},
        ])
        tool = CountingProgressTool("相同参数结果")

        result = run_to_result(Agent(model=model, tools=[tool]).run("repeat"))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "duplicate_tool_call")
        self.assertEqual(tool.calls, 2)
        self.assertEqual(model.calls, 3)
        self.assertIn("相同参数结果", result.final_answer)

    def test_changed_arguments_reset_duplicate_sequence(self):
        """验证参数变化后重复调用计数重新开始。"""
        model = ProgressSequenceModel([
            tool_response("call_1", "第一次", {"query": "same"}),
            tool_response("call_2", "参数变化", {"query": "changed"}),
            tool_response("call_3", "再次第一次", {"query": "same"}),
            {"text": "完成", "tool_calls": [], "usage": {}},
        ])
        tool = CountingProgressTool("结果")

        result = run_to_result(Agent(model=model, tools=[tool]).run("vary"))

        self.assertTrue(result.success)
        self.assertEqual(tool.calls, 3)
        self.assertEqual(model.calls, 4)

    def test_argument_order_does_not_bypass_duplicate_detection(self):
        """验证参数键顺序变化仍然被识别为相同调用。"""
        model = ProgressSequenceModel([
            tool_response("call_1", "第一次", {"first": 1, "second": 2}),
            tool_response("call_2", "第二次", {"second": 2, "first": 1}),
            tool_response("call_3", "第三次", {"first": 1, "second": 2}),
        ])
        tool = CountingProgressTool("规范化结果")

        result = run_to_result(Agent(model=model, tools=[tool]).run("order"))

        self.assertFalse(result.success)
        self.assertEqual(result.error, "duplicate_tool_call")
        self.assertEqual(tool.calls, 2)

    def test_stream_stops_before_third_identical_tool_call(self):
        """验证流式模式会发送重复调用错误终态。"""
        def make_response(call_id, text):
            def response():
                yield {"choices": [{"delta": {"content": text}}]}
                yield {"choices": [{"delta": {"tool_calls": [{
                    "index": 0,
                    "id": call_id,
                    "function": {
                        "name": "progress_tool",
                        "arguments": "{\"query\":\"same\"}",
                    },
                }]}}]}
            return response

        model = StreamingProgressModel([
            make_response("call_1", "一"),
            make_response("call_2", "二"),
            make_response("call_3", "三"),
        ])
        tool = CountingProgressTool("流式结果")

        events = list(Agent(model=model, tools=[tool]).run("stream repeat", stream=True))

        result = events[-1]["result"]
        self.assertEqual(events[-1]["type"], "done")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "duplicate_tool_call")
        self.assertEqual(tool.calls, 2)

    def test_compacts_large_tool_output_before_next_model_call(self):
        """验证超长工具输出进入下一次模型请求前会保留首尾并截断。"""
        model = ProgressSequenceModel([
            tool_response("call_1", "开始处理"),
            {"text": "已读取结果", "tool_calls": [], "usage": {}},
        ])
        tool = ProgressTool("A" * 12000 + "MIDDLE" + "Z" * 6000)

        result = run_to_result(Agent(model=model, tools=[tool]).run("inspect"))

        self.assertTrue(result.success)
        tool_messages = [
            message
            for message in model.message_snapshots[1]
            if message.get("role") == "tool"
        ]
        self.assertEqual(len(tool_messages), 1)
        compacted = tool_messages[0]["content"]
        self.assertLessEqual(len(compacted), Agent.TOOL_OUTPUT_MAX_CHARS)
        self.assertIn("tool output truncated", compacted)
        self.assertTrue(compacted.startswith("A"))
        self.assertTrue(compacted.endswith("Z"))
        self.assertNotIn("MIDDLE", compacted)

    def test_max_steps_returns_failed_result_with_partial_progress(self):
        """验证达到步数上限时返回失败状态和已产生的阶段性信息。"""
        model = ProgressSequenceModel([
            tool_response("call_1", "第一阶段已完成"),
            tool_response("call_2", "第二阶段正在处理"),
            {"text": "不得调用", "tool_calls": [], "usage": {}},
        ])
        tool = ProgressTool("工具已返回中间结果")

        result = run_to_result(
            Agent(model=model, tools=[tool], max_steps=2).run("long task")
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "max_steps_exceeded")
        self.assertEqual(result.step_count, 2)
        self.assertEqual(model.calls, 2)
        self.assertIn("第一阶段已完成", result.final_answer)
        self.assertIn("第二阶段正在处理", result.final_answer)

    def test_stream_max_steps_emits_failed_done_result_with_progress(self):
        """验证流式达到步数上限时发送失败的 done 结果。"""
        def tool_response():
            yield {"choices": [{"delta": {"content": "阶段性文字"}}]}
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {
                    "name": "progress_tool",
                    "arguments": "{}",
                },
            }]}}]}

        model = StreamingProgressModel([tool_response])
        tool = ProgressTool("流式工具结果")

        events = list(
            Agent(model=model, tools=[tool], max_steps=1).run(
                "stream task",
                stream=True,
            )
        )

        self.assertEqual([event["type"] for event in events], [
            "text", "tool_start", "tool_end", "done",
        ])
        result = events[-1]["result"]
        self.assertFalse(result.success)
        self.assertEqual(result.error, "max_steps_exceeded")
        self.assertIn("阶段性文字", result.final_answer)
        self.assertIn("流式工具结果", result.final_answer)


if __name__ == "__main__":
    unittest.main()
