import json
import unittest
from unittest.mock import MagicMock, patch

from cli import app as cli_app


class CliAppTest(unittest.TestCase):
    def test_parser_exposes_chat_web_and_config_commands(self):
        """验证统一入口声明三个可用子命令。"""
        parser = cli_app.build_parser()

        self.assertEqual(parser.parse_args(["chat"]).command, "chat")
        self.assertEqual(parser.parse_args(["web"]).command, "web")
        self.assertEqual(parser.parse_args(["config"]).command, "config")

    def test_main_uses_chat_for_default_and_explicit_command(self):
        """验证无子命令和 chat 都复用交互聊天入口。"""
        with patch.object(cli_app, "run_chat", return_value=7) as run_chat:
            self.assertEqual(cli_app.main([]), 7)
            self.assertEqual(cli_app.main(["chat"]), 7)

        self.assertEqual(run_chat.call_count, 2)

    def test_main_dispatches_web_and_config_commands(self):
        """验证 web 和 config 会分发到对应的启动函数。"""
        with (
            patch.object(cli_app, "run_web_server", return_value=3) as run_web,
            patch.object(cli_app, "show_config", return_value=4) as show_config,
        ):
            self.assertEqual(cli_app.main(["web"]), 3)
            self.assertEqual(cli_app.main(["config"]), 4)

        run_web.assert_called_once_with()
        show_config.assert_called_once_with()

    def test_show_config_masks_sensitive_values(self):
        """验证配置命令输出脱敏 JSON 而不是原始 API Key。"""
        output = MagicMock()

        exit_code = cli_app.show_config(
            load_config_func=MagicMock(),
            config_func=lambda: {
                "openai_api_key": "abcdefghijkl",
                "model": "openai",
            },
            output_func=output,
        )

        self.assertEqual(exit_code, 0)
        rendered = output.call_args.args[0]
        self.assertNotIn("abcdefghijkl", rendered)
        self.assertEqual(json.loads(rendered)["model"], "openai")

    def test_show_config_reports_loading_failure(self):
        """验证配置加载失败会返回失败码且输出安全说明。"""
        output = MagicMock()

        exit_code = cli_app.show_config(
            load_config_func=MagicMock(side_effect=ValueError("invalid config")),
            output_func=output,
        )

        self.assertEqual(exit_code, 1)
        output.assert_called_once_with("MyAgent 配置加载失败：invalid config")


if __name__ == "__main__":
    unittest.main()
