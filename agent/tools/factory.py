# encoding:utf-8
"""
工具工厂函数。
"""

from typing import List

from agent.tools.base import BaseTool
from agent.tools.bash import BashTool
from agent.tools.edit import EditTool
from agent.tools.ls import LsTool
from agent.tools.read import ReadTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from agent.tools.write import WriteTool


def create_default_tools() -> List[BaseTool]:
    """
    创建 Agent 默认基础工具列表。
    """
    return [
        BashTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        LsTool(),
        WebSearchTool(),
        WebFetchTool(),
    ]
