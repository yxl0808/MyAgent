import io
import logging
import tempfile
import unittest
from pathlib import Path

from common.console import print_safely
from common.logging import _HANDLER_MARKER, configure_logging
from common.paths import resolve_path


class CommonLoggingTest(unittest.TestCase):
    def setUp(self):
        """移除本测试创建的公共日志处理器并保存根日志级别。"""
        self.root_logger = logging.getLogger()
        self.previous_level = self.root_logger.level
        self._remove_common_handlers()

    def tearDown(self):
        """清理公共日志处理器并恢复根日志级别。"""
        self._remove_common_handlers()
        self.root_logger.setLevel(self.previous_level)

    def _remove_common_handlers(self):
        """移除带有 MyAgent 标记的日志处理器。"""
        for handler in list(self.root_logger.handlers):
            if getattr(handler, _HANDLER_MARKER, False):
                self.root_logger.removeHandler(handler)
                handler.close()

    def test_configure_logging_adds_one_handler_and_switches_debug(self):
        """验证日志初始化幂等且能切换 Debug 级别。"""
        output = io.StringIO()

        configure_logging(stream=output)
        configure_logging(debug=True, stream=io.StringIO())
        logging.getLogger("common-test").debug("debug message")

        handlers = [
            handler
            for handler in self.root_logger.handlers
            if getattr(handler, _HANDLER_MARKER, False)
        ]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(self.root_logger.level, logging.DEBUG)
        self.assertIn("debug message", output.getvalue())


class CommonPathsTest(unittest.TestCase):
    def test_resolve_path_expands_base_directory_and_returns_absolute_path(self):
        """验证路径工具正确处理相对路径、用户目录和绝对化。"""
        with tempfile.TemporaryDirectory(prefix="myagent_common_paths_") as temp_dir:
            resolved = resolve_path("nested/data", base_dir=temp_dir)

            self.assertEqual(resolved, (Path(temp_dir) / "nested/data").resolve())
            self.assertTrue(resolved.is_absolute())

    def test_resolve_path_keeps_absolute_path_independent_of_base_directory(self):
        """验证绝对路径不会被基目录重新拼接。"""
        with tempfile.TemporaryDirectory(prefix="myagent_common_paths_") as temp_dir:
            absolute = Path(temp_dir) / "absolute.txt"

            self.assertEqual(resolve_path(absolute, base_dir="other"), absolute.resolve())


class CommonConsoleTest(unittest.TestCase):
    def test_print_safely_replaces_characters_unsupported_by_output_encoding(self):
        """Verify console output handles unsupported characters."""
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="gbk", errors="strict")

        print_safely("Status: ✅", stream=stream)
        stream.flush()

        self.assertEqual(buffer.getvalue().decode("gbk").strip(), "Status: ?")


if __name__ == "__main__":
    unittest.main()
