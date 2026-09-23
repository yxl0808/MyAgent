import unittest
from unittest.mock import MagicMock, patch

from agent.agent import Agent
import main


class MainStartupTest(unittest.TestCase):
    def test_main_returns_failure_when_config_loading_fails(self):
        """验证配置加载失败时输出原因、返回失败码且不读取输入。"""
        with (
            patch.object(
                main,
                "load_config",
                side_effect=ValueError("invalid config"),
            ),
            patch.object(main, "create_agent") as create_agent,
            patch("builtins.input") as user_input,
            patch("cli.chat.print_safely") as output,
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 1)
        create_agent.assert_not_called()
        user_input.assert_not_called()
        output.assert_called_once_with("MyAgent 启动失败：invalid config")

    def test_main_returns_failure_when_agent_creation_fails(self):
        """验证 Agent 创建失败时输出原因、返回失败码且不读取输入。"""
        with (
            patch.object(main, "load_config"),
            patch.object(
                main,
                "create_agent",
                side_effect=ValueError("API key not configured"),
            ),
            patch("builtins.input") as user_input,
            patch("cli.chat.print_safely") as output,
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 1)
        user_input.assert_not_called()
        output.assert_called_once_with(
            "MyAgent 启动失败：API key not configured"
        )

    def test_main_prints_final_answer_from_non_stream_generator(self):
        """验证 CLI 能执行非流式 generator 并打印最终回答。"""
        model = MagicMock()
        model.chat.return_value = {
            'text': 'final answer',
            'tool_calls': [],
            'usage': {},
        }
        agent = Agent(model=model)

        with (
            patch.object(main, 'load_config'),
            patch.object(main, 'create_agent', return_value=agent),
            patch('builtins.input', side_effect=['hello', 'exit']),
            patch('cli.chat.print_safely') as mock_print,
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mock_print.assert_any_call('final answer')

    def test_resolve_agent_result_accepts_direct_result_and_generator(self):
        expected = MagicMock()

        def non_stream_run():
            if False:
                yield None
            return expected

        self.assertIs(main._resolve_agent_result(expected), expected)
        self.assertIs(main._resolve_agent_result(non_stream_run()), expected)

    def test_main_loads_config_before_creating_agent(self):
        events = []

        with (
            patch.object(
                main,
                'load_config',
                side_effect=lambda: events.append('load_config'),
            ),
            patch.object(
                main,
                'create_agent',
                side_effect=lambda: events.append('create_agent') or object(),
            ),
            patch('builtins.input', return_value='exit'),
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(events, ['load_config', 'create_agent'])


if __name__ == '__main__':
    unittest.main()
