# encoding:utf-8
"""发现并校验单个来源目录中的 Skill。"""

import re
from pathlib import Path

from agent.skills.frontmatter import FrontmatterError, parse_frontmatter
from agent.skills.types import Skill, SkillDiagnostic, SkillLoadResult
from common.paths import resolve_path


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RESERVED_NAMES = {"skills"}
_CONVENTIONAL_ENTRIES = {"SKILL.md", "scripts", "references", "assets"}


class SkillLoader:
    """扫描并校验一个内置或自定义 Skill 来源目录。"""

    def __init__(self, source_dir: Path, source: str):
        """记录规范化的来源目录和来源类型。"""
        self.source_dir = resolve_path(source_dir)
        self.source = source

    def load(self) -> SkillLoadResult:
        """加载当前来源中的一级 Skill，并隔离单个 Skill 的错误。"""
        if not self.source_dir.exists():
            return SkillLoadResult()
        if not self.source_dir.is_dir():
            return SkillLoadResult(
                diagnostics=(
                    self._error(
                        self.source_dir,
                        "Skill source is not a directory",
                    ),
                )
            )

        try:
            entries = sorted(
                self.source_dir.iterdir(),
                key=lambda item: item.name,
            )
        except OSError as exc:
            return SkillLoadResult(
                diagnostics=(
                    self._error(
                        self.source_dir,
                        f"Cannot read Skill source: {exc}",
                    ),
                )
            )

        skills = []
        diagnostics = []
        for entry in entries:
            if (
                entry.name.startswith(".")
                or entry.is_symlink()
                or not entry.is_dir()
            ):
                continue

            skill, entry_diagnostics = self._load_skill(entry)
            diagnostics.extend(entry_diagnostics)
            if skill is not None:
                skills.append(skill)

        return SkillLoadResult(
            skills=tuple(skills),
            diagnostics=tuple(diagnostics),
        )

    def _load_skill(self, skill_dir: Path):
        """读取并校验一个候选目录，返回 Skill 和对应诊断。"""
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            return None, [
                self._error(
                    skill_dir,
                    "Skill directory has no readable SKILL.md",
                )
            ]

        try:
            content = skill_file.read_text(encoding="utf-8-sig")
            parsed = parse_frontmatter(content)
        except (OSError, UnicodeError, FrontmatterError) as exc:
            return None, [
                self._error(
                    skill_file,
                    str(exc),
                    skill_dir.name,
                )
            ]

        diagnostics = [
            SkillDiagnostic(
                severity="warning",
                source=self.source,
                path=str(skill_file.resolve()),
                message=warning,
                skill_name=parsed.name,
            )
            for warning in parsed.warnings
        ]

        if not _NAME_RE.fullmatch(parsed.name):
            diagnostics.append(
                self._error(
                    skill_file,
                    f'Invalid Skill name "{parsed.name}"; '
                    "use lowercase hyphen-case",
                    parsed.name,
                )
            )
            return None, diagnostics

        if parsed.name in _RESERVED_NAMES:
            diagnostics.append(
                self._error(
                    skill_file,
                    f'Reserved Skill name: "{parsed.name}"',
                    parsed.name,
                )
            )
            return None, diagnostics

        if skill_dir.name != parsed.name:
            diagnostics.append(
                self._error(
                    skill_file,
                    f'Skill name "{parsed.name}" does not match '
                    f'directory "{skill_dir.name}"',
                    parsed.name,
                )
            )
            return None, diagnostics

        for child in sorted(
            skill_dir.iterdir(),
            key=lambda item: item.name,
        ):
            if child.name not in _CONVENTIONAL_ENTRIES:
                diagnostics.append(
                    SkillDiagnostic(
                        severity="warning",
                        source=self.source,
                        path=str(child.resolve()),
                        message=(
                            "Non-conventional auxiliary entry; MyAgent "
                            "preserves but does not validate it."
                        ),
                        skill_name=parsed.name,
                    )
                )

        skill = Skill(
            name=parsed.name,
            description=parsed.description,
            source=self.source,
            skill_dir=str(skill_dir.resolve()),
            skill_file=str(skill_file.resolve()),
            body=parsed.body,
        )
        return skill, diagnostics

    def _error(self, path, message, skill_name=None):
        """为当前来源构造统一格式的错误诊断。"""
        return SkillDiagnostic(
            severity="error",
            source=self.source,
            path=str(Path(path).resolve()),
            message=message,
            skill_name=skill_name,
        )
