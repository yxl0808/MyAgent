# encoding:utf-8
"""
Agent 工厂函数。
"""

from typing import Optional

from agent.agent import Agent
from agent.memory import create_memory_manager
from agent.skills.factory import create_skill_manager
from agent.tools.base import BaseTool
from agent.tools.factory import create_default_tools
from agent.tools.memory import (
    MemoryFlushTool,
    MemoryGetTool,
    MemoryRebuildTool,
    MemoryRefineTool,
    MemorySearchTool,
    MemoryStatusTool,
)
from models import create_model


def create_agent(
    model=None,
    system_prompt: str = "",
    tools: Optional[list[BaseTool]] = None,
    workspace_dir: str = "",
) -> Agent:
    """
    创建 Agent，并根据配置决定是否启用长期记忆。
    """
    from config import conf

    if model is None:
        model = create_model()

    effective_workspace = workspace_dir or conf().get(
        "agent_workspace",
        "~/myagent",
    )
    agent_enabled = conf().get("agent_enabled", True)
    memory_enabled = conf().get("memory_enabled", True)
    memory_manager = (
        create_memory_manager(
            llm_model=model,
            workspace_dir=effective_workspace,
        )
        if memory_enabled
        else None
    )
    skills_enabled = conf().get("skills_enabled", True)
    skill_manager = (
        create_skill_manager(
            workspace_dir=effective_workspace,
            builtin_dir=conf().get("skills_builtin_dir", ""),
        )
        if skills_enabled
        else None
    )
    agent_tools = (
        list(tools) if tools is not None else create_default_tools()
    ) if agent_enabled else []
    if agent_enabled and memory_enabled:
        registered_tool_names = {tool.name for tool in agent_tools}
        for memory_tool in (
            MemorySearchTool(),
            MemoryGetTool(),
            MemoryFlushTool(),
            MemoryRefineTool(),
            MemoryRebuildTool(),
            MemoryStatusTool(),
        ):
            if memory_tool.name not in registered_tool_names:
                agent_tools.append(memory_tool)
                registered_tool_names.add(memory_tool.name)

    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=agent_tools,
        max_steps=conf().get("agent_max_steps", 20),
        workspace_dir=effective_workspace,
        memory_manager=memory_manager,
        skill_manager=skill_manager,
    )
