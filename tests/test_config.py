import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config as config_module
from config import _parse_env_value


class ConfigEnvParsingTest(unittest.TestCase):
    def test_load_config_applies_safe_environment_overrides(self):
        """验证 load_config 通过安全解析应用环境变量覆盖。"""
        previous_config = config_module.config
        with tempfile.TemporaryDirectory(prefix="myagent_config_test_") as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "agent_enabled": True,
                    "agent_max_steps": 20,
                    "openai_api_key": "file-key",
                }),
                encoding="utf-8",
            )
            environment = {
                "AGENT_ENABLED": "false",
                "AGENT_MAX_STEPS": "7",
                "OPENAI_API_KEY": "123456",
            }

            try:
                with (
                    patch.object(config_module, "get_root", return_value=temp_dir),
                    patch.dict(os.environ, environment, clear=True),
                ):
                    config_module.load_config()

                self.assertIs(config_module.conf().get("agent_enabled"), False)
                self.assertEqual(config_module.conf().get("agent_max_steps"), 7)
                self.assertEqual(config_module.conf().get("openai_api_key"), "123456")
            finally:
                config_module.config = previous_config

    def test_load_config_parses_web_show_thinking_override(self):
        """验证 Web 思考显示开关使用布尔环境变量解析。"""
        previous_config = config_module.config
        with tempfile.TemporaryDirectory(prefix="myagent_web_config_test_") as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            try:
                with (
                    patch.object(config_module, "get_root", return_value=temp_dir),
                    patch.dict(
                        os.environ,
                        {"WEB_SHOW_THINKING": "true"},
                        clear=True,
                    ),
                ):
                    config_module.load_config()

                self.assertIs(
                    config_module.conf().get("web_show_thinking"),
                    True,
                )
            finally:
                config_module.config = previous_config

    def test_parse_env_value_uses_declared_config_types(self):
        """验证环境变量只按配置声明类型进行安全转换。"""
        self.assertIs(_parse_env_value("agent_enabled", " true "), True)
        self.assertIs(_parse_env_value("memory_enabled", "FALSE"), False)
        self.assertEqual(_parse_env_value("agent_max_steps", " 7 "), 7)
        self.assertEqual(_parse_env_value("agent_temperature", " 0.25 "), 0.25)
        self.assertEqual(_parse_env_value("openai_api_key", "123456"), "123456")
        self.assertEqual(_parse_env_value("openai_model", "True"), "True")

    def test_parse_env_value_never_executes_python_expressions(self):
        """验证恶意 Python 表达式只作为普通字符串保留。"""
        expression = '__import__("os").system("unsafe")'

        parsed = _parse_env_value("openai_api_key", expression)

        self.assertEqual(parsed, expression)

    def test_parse_env_value_preserves_invalid_typed_values(self):
        """验证无法按声明类型转换的值保持原字符串。"""
        self.assertEqual(_parse_env_value("agent_enabled", "yes"), "yes")
        self.assertEqual(_parse_env_value("agent_max_steps", "many"), "many")
        self.assertEqual(_parse_env_value("agent_top_p", "wide"), "wide")
        self.assertEqual(_parse_env_value("unknown_setting", "42"), "42")


if __name__ == "__main__":
    unittest.main()
