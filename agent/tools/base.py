# encoding:utf-8
"""
MyAgent 工具系统 — 基类模块
===========================
定义所有工具的抽象基类 BaseTool。

一个工具就是 Agent 的"手脚"——LLM 只能思考，不能行动。
工具让 Agent 能执行系统命令、读写文件、搜索网络、检索记忆。

工具的四要素（每个子类必须提供）:
  name:        工具名（LLM 通过名字调用），如 "bash" / "web_search"
  description: 工具描述（告诉 LLM 这个工具是干什么的）
  parameters:  参数定义（JSON Schema 格式，告诉 LLM 调用时需要传什么参数）
  execute():   执行逻辑（子类实现，接收参数 dict，返回执行结果）
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    所有工具的抽象基类。

    子类只需要做三件事:
      1. 设置 name / description / parameters 类属性
      2. 实现 execute(params) 方法

    示例:
        class BashTool(BaseTool):
            name = "bash"
            description = "执行 shell 命令"
            parameters = {
                "command": {
                    "type": "string",
                    "description": "要执行的命令",
                    "required": True,
                }
            }

            def execute(self, params):
                return subprocess.run(params["command"], shell=True, capture_output=True).stdout
    """

    # ── 子类必须覆写的属性 ────────────────────────────────────
    name: str = ""           # 工具名，如 "bash"
    description: str = ""    # 工具描述，LLM 据此判断何时使用
    parameters: Dict[str, Any] = {}  # JSON Schema 参数定义

    def execute(self, params: Dict[str, Any]) -> Any:
        """
        执行工具逻辑。子类必须覆写此方法。

        入参:
            params: 工具参数 dict，key 和 parameters 中定义的属性名对应
                    如 {"command": "ls -la"}

        返回:
            工具执行结果，类型不限（字符串/字典/列表均可）
            返回的内容会被放入对话历史发给 LLM
        """
        raise NotImplementedError(
            f"Tool '{self.name}' must implement execute() method"
        )

    def to_definition(self) -> Dict[str, Any]:
        """
        将工具转换为 LLM API 期望的工具定义格式。

        返回格式（Claude/统一格式）:
            {
                "name": "bash",
                "description": "执行 shell 命令",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的命令"}
                    },
                    "required": ["command"]
                }
            }

        模型适配器会自动将此格式转换为各 API 的原生格式
        （OpenAI 的 function.parameters / Claude 的 input_schema）。
        """
        # 从 parameters 中提取 required 列表
        required = [
            name for name, info in self.parameters.items()
            if info.get("required", False)
        ]

        properties = {
            parameter_name: {
                key: value
                for key, value in parameter_info.items()
                if key != "required"
            }
            for parameter_name, parameter_info in self.parameters.items()
        }

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _get_agent(self):
        """返回注册该工具的 Agent 引用。"""
        return getattr(self, "_agent", None)

    def _get_workspace_dir(self) -> Optional[str]:
        """返回当前 Agent 的工作区绝对路径。"""
        agent = self._get_agent()
        workspace = getattr(agent, "workspace_dir", "") if agent else ""
        if not workspace:
            return None
        return str(Path(workspace).expanduser().resolve())

    def _resolve_path(self, path: str) -> str:
        """将相对路径解析到 Agent 工作区，绝对路径保持可访问。"""
        target = Path(path).expanduser()
        if target.is_absolute():
            return str(target.resolve())

        workspace = self._get_workspace_dir()
        base = Path(workspace) if workspace else Path.cwd()
        return str((base / target).resolve())
