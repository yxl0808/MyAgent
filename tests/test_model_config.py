import unittest
from unittest.mock import MagicMock, patch

from models import create_model
from models.claude import ClaudeAdapter
from models.deepseek import DeepSeekAdapter
from models.openai import OpenAIAdapter


class ModelConfigTest(unittest.TestCase):
    def test_create_model_passes_configured_top_p_to_adapter(self):
        """验证模型工厂会把 agent_top_p 传给适配器。"""
        config = {
            "openai_api_key": "test-key",
            "openai_api_base": "https://example.com/v1",
            "openai_model": "test-model",
            "proxy": "",
            "agent_temperature": 0.4,
            "agent_top_p": 0.6,
            "request_timeout": 45,
        }

        with patch("models.conf", return_value=config):
            model = create_model("openai")

        self.assertEqual(
            model.config,
            {
                "temperature": 0.4,
                "top_p": 0.6,
                "request_timeout": 45,
            },
        )

    def test_chat_uses_configured_timeout_for_sync_and_stream_requests(self):
        """验证同步与流式模型请求都使用配置的超时时间。"""
        model = OpenAIAdapter(
            api_key="test-key",
            api_base="https://example.com/v1",
            model="test-model",
            config={"request_timeout": 45},
        )
        sync_response = MagicMock()
        sync_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "done", "tool_calls": []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        stream_response = MagicMock()
        stream_response.iter_content.return_value = []

        with patch.object(
            model,
            "_http_post",
            side_effect=[sync_response, stream_response],
        ) as http_post:
            sync_result = model.chat([{"role": "user", "content": "hello"}])
            stream_result = model.chat(
                [{"role": "user", "content": "hello"}],
                stream=True,
            )
            stream_chunks = list(stream_result)

        self.assertIsInstance(sync_result, dict)
        self.assertEqual(sync_result["text"], "done")
        self.assertEqual(stream_chunks, [])
        self.assertEqual(http_post.call_count, 2)
        self.assertEqual(
            http_post.call_args_list[0].args[0],
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(http_post.call_args_list[0].kwargs["timeout"], 45)
        self.assertFalse(http_post.call_args_list[0].kwargs["stream"])
        self.assertEqual(http_post.call_args_list[1].kwargs["timeout"], 45)
        self.assertTrue(http_post.call_args_list[1].kwargs["stream"])

    def test_openai_and_claude_request_bodies_include_top_p(self):
        """验证 OpenAI 和 Claude 请求体都会包含配置的 top_p。"""
        openai = OpenAIAdapter(
            api_key="test-key",
            api_base="https://example.com/v1",
            model="test-model",
            config={"top_p": 0.6},
        )
        claude = ClaudeAdapter(
            api_key="test-key",
            api_base="https://example.com/v1",
            model="test-model",
            config={"top_p": 0.6},
        )
        messages = [{"role": "user", "content": "hello"}]

        openai_body = openai._build_request_body(messages, None, False)
        claude_body = claude._build_request_body(messages, None, False)

        self.assertEqual(openai_body["top_p"], 0.6)
        self.assertEqual(claude_body["top_p"], 0.6)

    def test_deepseek_thinking_removes_top_p(self):
        """验证 DeepSeek thinking 模式会移除不生效的 top_p。"""
        deepseek = DeepSeekAdapter(
            api_key="test-key",
            api_base="https://example.com/v1",
            model="deepseek-v4",
            config={"top_p": 0.6},
        )

        body = deepseek._build_request_body(
            [{"role": "user", "content": "hello"}],
            None,
            False,
        )

        self.assertNotIn("top_p", body)
        self.assertEqual(body["thinking"], {"type": "enabled"})


if __name__ == "__main__":
    unittest.main()
