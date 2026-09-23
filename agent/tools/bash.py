# encoding:utf-8
"""
MyAgent 工具 — Bash 执行器
==========================
允许 Agent 执行 shell 命令并获取输出。
这是 Agent 最基础也是最强大的工具——通过它，Agent 可以操作整个操作系统。

安全注意事项:
  - 执行的命令在 Agent 宿主机器上运行，具有当前用户的全部权限
  - 不提供沙箱隔离——Agent 能做的事情取决于运行它的用户权限
  - 在生产环境建议用 Docker 容器运行，限制影响范围
  - 命令有超时限制（默认 120 秒），防止死循环
"""

import logging
import subprocess
from typing import Any, Dict

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class BashTool(BaseTool):
    """
    执行 shell 命令并返回标准输出。

    Agent 通过此工具可以:
      - 运行 Python 脚本
      - 安装 pip 包
      - 操作文件系统
      - 调用系统命令

    参数:
        command: 要执行的 shell 命令字符串，如 "ls -la" / "python hello.py"
    """

    name = "bash"
    description = (
        "Execute a shell command and return its output. "
        "Use this to run code, install packages, manage files, or perform any system operation. "
        "Commands run in the agent's workspace directory. "
        "The output (stdout) is returned; if the command fails, stderr is included in the result."
    )
    parameters = {
        "command": {
            "type": "string",
            "description": "The shell command to execute, e.g. 'ls -la' or 'python script.py'",
            "required": True,
        }
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        执行指定的 shell 命令。

        执行过程:
          1. 从 params 中取出 command 字符串
          2. 用 subprocess.run 在 shell 中执行
          3. 捕获 stdout 和 stderr
          4. 如果命令成功（返回码 0），返回 stdout
          5. 如果命令失败，返回 stdout + stderr（含错误信息）

        超时: 默认 120 秒，防止命令卡死导致 Agent 永久等待

        入参:
            params: {"command": "ls -la"}

        返回:
            命令执行的输出字符串（stdout 或 stdout + stderr）
        """
        command = params.get("command", "").strip()
        if not command:
            return "Error: No command provided"

        logger.info("[Bash] Executing: %s", command[:200])

        try:
            # 执行命令
            # shell=True: 通过 shell 执行（支持管道、重定向等 shell 特性）
            # capture_output=True: 捕获 stdout 和 stderr（不打印到终端）
            # text=True: 输出以字符串形式返回（而非 bytes）
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,  # 120 秒超时，防止无限循环
                cwd=self._get_workspace_dir(),
            )

            # ── 收集输出 ──
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[stderr]\n{stderr}")

            final_output = "\n".join(output_parts) if output_parts else "(no output)"

            # 记录执行结果
            if result.returncode == 0:
                logger.info("[Bash] Success, output: %s", final_output[:200])
            else:
                logger.warning("[Bash] Exit code %d: %s", result.returncode, final_output[:200])
                final_output = f"[exit code {result.returncode}]\n{final_output}"

            return final_output

        except subprocess.TimeoutExpired:
            logger.error("[Bash] Command timed out after 120s: %s", command[:200])
            return f"Error: Command timed out after 120 seconds. Try a simpler command or split into smaller steps."
        except Exception as e:
            logger.error("[Bash] Execution failed: %s", e)
            return f"Error executing command: {e}"