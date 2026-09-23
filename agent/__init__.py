# encoding:utf-8
"""
MyAgent Agent 核心模块
======================
导出 Agent 核心类和数据类型，供外部模块使用。

用法:
    from agent import Agent, AgentResult, AgentAction, AgentActionType

    agent = Agent(model=llm, system_prompt="...", tools=[...])
    result = agent.run("你好")
"""

from agent.agent import Agent
from agent.factory import create_agent
from agent.types import AgentAction, AgentActionType, AgentResult, ToolResult

__all__ = [
    "Agent",
    "create_agent",
    "AgentAction",
    "AgentActionType",
    "AgentResult",
    "ToolResult",
]
