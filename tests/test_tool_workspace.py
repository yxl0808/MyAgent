import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.tools.bash import BashTool
from agent.tools.edit import EditTool
from agent.tools.ls import LsTool
from agent.tools.read import ReadTool
from agent.tools.write import WriteTool


class ToolWorkspaceTest(unittest.TestCase):
    def setUp(self):
        """创建每个测试独立使用的临时工作区。"""
        self.temp_dir = tempfile.TemporaryDirectory(prefix="myagent_tool_workspace_")
        self.workspace = Path(self.temp_dir.name).resolve()
        self.agent = SimpleNamespace(workspace_dir=str(self.workspace))

    def tearDown(self):
        """清理临时工作区。"""
        self.temp_dir.cleanup()

    def _register(self, tool):
        """把工具绑定到带工作区的测试 Agent。"""
        tool._agent = self.agent
        return tool

    def test_file_tools_resolve_relative_paths_from_workspace(self):
        """验证读写编辑工具的相对路径统一基于 Agent 工作区。"""
        write_tool = self._register(WriteTool())
        read_tool = self._register(ReadTool())
        edit_tool = self._register(EditTool())

        write_tool.execute({"path": "notes/item.txt", "content": "before"})
        read_output = read_tool.execute({"path": "notes/item.txt"})
        edit_output = edit_tool.execute({
            "path": "notes/item.txt",
            "old_string": "before",
            "new_string": "after",
        })

        target = self.workspace / "notes" / "item.txt"
        self.assertTrue(target.exists())
        self.assertIn("before", read_output)
        self.assertEqual(target.read_text(encoding="utf-8"), "after")
        self.assertTrue(Path(f"{target}.bak").exists())
        self.assertIn(str(target), edit_output)

    def test_ls_defaults_to_workspace_root(self):
        """验证 ls 未传路径时列出 Agent 工作区根目录。"""
        (self.workspace / "example.txt").write_text("content", encoding="utf-8")
        ls_tool = self._register(LsTool())

        output = ls_tool.execute({})

        self.assertIn(f"Directory: {self.workspace}", output)
        self.assertIn("example.txt", output)

    def test_absolute_paths_remain_accessible_outside_workspace(self):
        """验证显式绝对路径不受工作区限制。"""
        with tempfile.TemporaryDirectory(prefix="myagent_absolute_path_") as outside:
            target = Path(outside).resolve() / "outside.txt"
            write_tool = self._register(WriteTool())
            read_tool = self._register(ReadTool())

            write_tool.execute({"path": str(target), "content": "outside"})
            output = read_tool.execute({"path": str(target)})

            self.assertTrue(target.exists())
            self.assertIn("outside", output)

    def test_bash_uses_agent_workspace_as_cwd(self):
        """验证 Bash 子进程使用 Agent 工作区作为 cwd。"""
        bash_tool = self._register(BashTool())
        completed = MagicMock(returncode=0, stdout="done", stderr="")

        with patch("agent.tools.bash.subprocess.run", return_value=completed) as run:
            output = bash_tool.execute({"command": "test-command"})

        self.assertEqual(output, "done")
        self.assertEqual(run.call_args.kwargs["cwd"], str(self.workspace))

    def test_tools_without_agent_fall_back_to_process_context(self):
        """验证未注册 Agent 时路径和 Bash 保持进程上下文行为。"""
        read_tool = ReadTool()
        bash_tool = BashTool()
        completed = MagicMock(returncode=0, stdout="done", stderr="")

        self.assertEqual(read_tool._get_workspace_dir(), None)
        self.assertEqual(read_tool._resolve_path("relative.txt"), str((Path.cwd() / "relative.txt").resolve()))
        with patch("agent.tools.bash.subprocess.run", return_value=completed) as run:
            bash_tool.execute({"command": "test-command"})

        self.assertIsNone(run.call_args.kwargs["cwd"])


if __name__ == "__main__":
    unittest.main()
