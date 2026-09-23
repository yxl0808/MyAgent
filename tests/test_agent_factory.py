import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.factory import create_agent
from agent.tools.memory import MemorySearchTool


class AgentFactoryTest(unittest.TestCase):
    def test_create_agent_disables_tools_but_keeps_memory_in_chat_mode(self):
        """验证关闭 Agent 模式时禁用全部工具，但保留长期记忆。"""
        memory_manager = SimpleNamespace()
        manual_tool = MemorySearchTool()

        with (
            patch(
                "config.conf",
                return_value={
                    "agent_enabled": False,
                    "memory_enabled": True,
                    "agent_workspace": "configured-workspace",
                    "agent_max_steps": 20,
                },
            ),
            patch(
                "agent.factory.create_memory_manager",
                return_value=memory_manager,
            ) as create_memory_manager,
        ):
            agent = create_agent(model=object(), tools=[manual_tool])

        self.assertEqual(agent.tools, [])
        self.assertIs(agent.memory_manager, memory_manager)
        create_memory_manager.assert_called_once_with(
            llm_model=agent.model,
            workspace_dir="configured-workspace",
        )
        self.assertIsNone(manual_tool._get_agent())

    def test_create_agent_uses_configured_workspace_when_not_explicit(self):
        """验证未显式传入工作区时回退到 agent_workspace 配置。"""
        memory_manager = SimpleNamespace()

        with (
            patch(
                "config.conf",
                return_value={
                    "agent_workspace": "configured-workspace",
                    "memory_enabled": True,
                    "agent_max_steps": 20,
                },
            ),
            patch(
                "agent.factory.create_memory_manager",
                return_value=memory_manager,
            ) as create_memory_manager,
        ):
            agent = create_agent(model=object(), tools=[])

        self.assertEqual(agent.workspace_dir, "configured-workspace")
        create_memory_manager.assert_called_once_with(
            llm_model=agent.model,
            workspace_dir="configured-workspace",
        )

    def test_create_agent_uses_explicit_workspace_for_agent_and_memory(self):
        """验证显式工作区优先并同时传给 Agent 和记忆工厂。"""
        memory_manager = SimpleNamespace()

        with (
            patch(
                "config.conf",
                return_value={
                    "agent_workspace": "configured-workspace",
                    "memory_enabled": True,
                    "agent_max_steps": 20,
                },
            ),
            patch(
                "agent.factory.create_memory_manager",
                return_value=memory_manager,
            ) as create_memory_manager,
        ):
            agent = create_agent(
                model=object(),
                tools=[],
                workspace_dir="explicit-workspace",
            )

        self.assertEqual(agent.workspace_dir, "explicit-workspace")
        create_memory_manager.assert_called_once_with(
            llm_model=agent.model,
            workspace_dir="explicit-workspace",
        )

    def test_create_agent_disables_default_memory_but_keeps_manual_tools(self):
        """验证关闭记忆时不初始化默认记忆能力，并保留手动工具。"""
        manual_memory_tool = MemorySearchTool()

        with (
            patch(
                "config.conf",
                return_value={"memory_enabled": False, "agent_max_steps": 20},
            ),
            patch("agent.factory.create_memory_manager") as create_memory_manager,
        ):
            agent = create_agent(model=object(), tools=[manual_memory_tool])

        create_memory_manager.assert_not_called()
        self.assertIsNone(agent.memory_manager)
        self.assertEqual(agent.tools, [manual_memory_tool])
        self.assertIs(manual_memory_tool._get_agent(), agent)

    def test_create_agent_uses_configured_max_steps(self):
        """验证工厂会把 agent_max_steps 配置传给 Agent。"""
        memory_manager = SimpleNamespace()

        with (
            patch("config.conf", return_value={"agent_max_steps": 7}),
            patch("agent.factory.create_memory_manager", return_value=memory_manager),
        ):
            agent = create_agent(model=object(), tools=[])

        self.assertEqual(agent.max_steps, 7)

    def test_create_agent_registers_all_memory_tools(self):
        class DummyModel:
            pass

        memory_manager = SimpleNamespace()

        with patch("agent.factory.create_memory_manager", return_value=memory_manager):
            agent = create_agent(model=DummyModel(), tools=[])

        tool_names = {tool.name for tool in agent.tools}
        self.assertTrue(
            {
                "memory_search",
                "memory_get",
                "memory_rebuild_vectors",
                "memory_status",
                "memory_flush",
                "memory_refine",
            }.issubset(tool_names)
        )
        for tool in agent.tools:
            self.assertIs(getattr(tool, "_agent", None), agent)

    def test_create_agent_builds_skill_manager_with_effective_workspace(self):
        """验证工厂使用有效工作区和配置目录创建 SkillManager。"""
        skill_manager = SimpleNamespace()

        with (
            patch(
                "config.conf",
                return_value={
                    "agent_workspace": "configured-workspace",
                    "memory_enabled": False,
                    "skills_enabled": True,
                    "skills_builtin_dir": "configured-skills",
                    "agent_max_steps": 20,
                },
            ),
            patch("agent.factory.create_memory_manager") as create_memory_manager,
            patch(
                "agent.factory.create_skill_manager",
                return_value=skill_manager,
            ) as create_skill_manager,
        ):
            agent = create_agent(model=object(), tools=[])

        create_memory_manager.assert_not_called()
        create_skill_manager.assert_called_once_with(
            workspace_dir="configured-workspace",
            builtin_dir="configured-skills",
        )
        self.assertIs(agent.skill_manager, skill_manager)

    def test_create_agent_does_not_build_skill_manager_when_disabled(self):
        """验证关闭 Skill 功能时工厂不创建管理器。"""
        with (
            patch(
                "config.conf",
                return_value={
                    "memory_enabled": False,
                    "skills_enabled": False,
                    "agent_max_steps": 20,
                },
            ),
            patch("agent.factory.create_skill_manager") as create_skill_manager,
        ):
            agent = create_agent(model=object(), tools=[])

        create_skill_manager.assert_not_called()
        self.assertIsNone(agent.skill_manager)


if __name__ == "__main__":
    unittest.main()
