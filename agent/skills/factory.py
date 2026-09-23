# encoding:utf-8
"""根据项目和工作区路径创建 SkillManager。"""

from pathlib import Path

from agent.skills.manager import SkillManager
from common.paths import resolve_path
from config import get_root


def create_skill_manager(workspace_dir: str, builtin_dir: str = ""):
    """解析 Skill 来源的绝对路径并创建 SkillManager。"""
    project_root = resolve_path(get_root())

    if builtin_dir:
        resolved_builtin = resolve_path(builtin_dir, base_dir=project_root)
    else:
        resolved_builtin = project_root / "skills"

    resolved_workspace = resolve_path(workspace_dir, base_dir=project_root)

    return SkillManager(
        builtin_dir=resolved_builtin,
        custom_dir=(resolved_workspace / "skills"),
    )
