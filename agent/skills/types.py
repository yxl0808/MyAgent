# encoding:utf-8
"""Skill 系统的数据类型定义。"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class ParsedFrontmatter:
    """保存从 SKILL.md 解析出的最小元数据和正文。"""

    name: str
    description: str
    body: str
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Skill:
    """表示一个已经校验、可被显式调用的本地 Skill。"""

    name: str
    description: str
    source: str
    skill_dir: str
    skill_file: str
    body: str


@dataclass(frozen=True)
class SkillDiagnostic:
    """描述 Skill 加载过程中产生的一条警告或错误。"""

    severity: str
    source: str
    path: str
    message: str
    skill_name: Optional[str] = None


@dataclass(frozen=True)
class SkillLoadResult:
    """汇总单个来源的有效 Skill 和加载诊断。"""

    skills: Tuple[Skill, ...] = ()
    diagnostics: Tuple[SkillDiagnostic, ...] = ()


@dataclass(frozen=True)
class SkillSnapshot:
    """保存一次刷新后对外可见的不可变 Skill 快照。"""

    skills: Mapping[str, Skill] = field(
        default_factory=lambda: MappingProxyType({})
    )
    diagnostics: Tuple[SkillDiagnostic, ...] = ()

    @classmethod
    def create(cls, skills, diagnostics=()):
        """复制映射和诊断，构造不会被调用方修改的快照。"""
        return cls(
            skills=MappingProxyType(dict(skills)),
            diagnostics=tuple(diagnostics),
        )
