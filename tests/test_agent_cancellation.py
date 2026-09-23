import threading
import unittest

from agent.agent import Agent
from agent.tools.base import BaseTool


class StreamingSequenceModel:
    """返回确定性 OpenAI 风格流式响应的测试模型。"""

    def __init__(self, factories):
        """保存流式响应工厂和模型调用次数。"""
        self.factories = list(factories)
        self.calls = 0

    def chat(self, messages, tools=None, stream=False):
        """返回下一个确定性响应生成器。"""
        self.calls += 1
        return self.factories.pop(0)()


class CancellingTool(BaseTool):
    """在确定性工具执行期间设置取消信号。"""

    name = "cancel_tool"
    description = "Cancel the test run"
    parameters = {}

    def __init__(self, cancel_event):
        """保存测试使用的取消信号。"""
        self.cancel_event = cancel_event

    def execute(self, params):
        """请求取消并返回已完成的工具结果。"""
        self.cancel_event.set()
        return "tool finished"


class AgentCancellationTest(unittest.TestCase):
    def test_preset_cancellation_skips_model_call(self):
        """验证预先取消不会调用模型。"""
        model = StreamingSequenceModel([])
        event = threading.Event()
        event.set()

        events = list(Agent(model=model).run("hello", stream=True, cancel_event=event))

        self.assertEqual(model.calls, 0)
        self.assertEqual([item["type"] for item in events], ["stopped"])

    def test_stream_cancellation_keeps_first_chunk_and_omits_done(self):
        """验证分块间取消保留已输出文字并以 stopped 结束。"""
        event = threading.Event()

        def stream_response():
            yield {"choices": [{"delta": {"content": "A"}}]}
            event.set()
            yield {"choices": [{"delta": {"content": "B"}}]}

        model = StreamingSequenceModel([stream_response])
        events = list(Agent(model=model).run("hello", stream=True, cancel_event=event))

        self.assertEqual([item["type"] for item in events], ["text", "stopped"])
        self.assertEqual(events[0]["content"], "A")
        self.assertEqual(events[1]["result"].final_answer, "A")

    def test_tool_cancellation_stops_before_second_model_call(self):
        """验证工具返回后的取消阻止下一次模型调用。"""
        event = threading.Event()

        def tool_response():
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "cancel_tool",
                                "arguments": "{}",
                            },
                        }]
                    }
                }]
            }

        model = StreamingSequenceModel([tool_response])
        agent = Agent(model=model, tools=[CancellingTool(event)])

        events = list(agent.run("use tool", stream=True, cancel_event=event))

        self.assertEqual(model.calls, 1)
        self.assertEqual(
            [item["type"] for item in events],
            ["tool_start", "tool_end", "stopped"],
        )

    def test_run_without_cancel_event_still_finishes(self):
        """验证未传取消信号时保留原有 done 事件。"""
        def final_response():
            yield {"choices": [{"delta": {"content": "done"}}]}

        events = list(Agent(
            model=StreamingSequenceModel([final_response])
        ).run("hello", stream=True))

        self.assertEqual([item["type"] for item in events], ["text", "done"])


if __name__ == "__main__":
    unittest.main()
