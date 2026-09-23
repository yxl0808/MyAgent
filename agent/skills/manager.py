# encoding:utf-8
"""合并 Skill 来源并维护对外可见的 Skill 快照。"""

from pathlib import Path

from agent.skills.loader import SkillLoader
from agent.skills.types import SkillDiagnostic, SkillSnapshot
from common.paths import resolve_path


class SkillRefreshError(RuntimeError):
    """表示 Skill 快照因内部异常无法完成刷新。"""


class SkillManager:
    """合并 Skill 来源，并提供刷新、查询和诊断展示。"""

    def __init__(self, builtin_dir, custom_dir):
        """记录内置与自定义目录，并创建空的初始快照。"""
        self.builtin_dir = resolve_path(builtin_dir)
        self.custom_dir = resolve_path(custom_dir)
        self._snapshot = SkillSnapshot.create({})

    @property
    def snapshot(self):
        """返回最近一次成功刷新的不可变 Skill 快照。"""
        return self._snapshot

    def refresh(self):
        """重新加载两个来源，应用冲突规则后原子替换快照。"""
        try:
            builtin_result = SkillLoader(
                self.builtin_dir,
                "builtin",
            ).load()
            custom_result = SkillLoader(
                self.custom_dir,
                "custom",
            ).load()

            diagnostics = list(builtin_result.diagnostics)
            skills = self._deduplicate_source(
                builtin_result.skills,
                "builtin",
                diagnostics,
            )
            custom_skills = self._deduplicate_source(
                custom_result.skills,
                "custom",
                diagnostics,
            )
            diagnostics.extend(custom_result.diagnostics)

            for name, custom_skill in custom_skills.items():
                if name in skills:
                    diagnostics.append(
                        SkillDiagnostic(
                            severity="error",
                            source="custom",
                            path=custom_skill.skill_file,
                            message=(
                                f'Custom Skill "{name}" conflicts with '
                                "builtin Skill; the custom Skill was not loaded."
                            ),
                            skill_name=name,
                        )
                    )
                    continue

                skills[name] = custom_skill

            new_snapshot = SkillSnapshot.create(
                skills,
                diagnostics,
            )
        except Exception as exc:
            raise SkillRefreshError(
                f"Skill refresh failed: {exc}"
            ) from exc

        self._snapshot = new_snapshot
        return self._snapshot

    def get_skill(self, name):
        """从当前快照按规范名称查询可用 Skill。"""
        return self._snapshot.skills.get(name)

    def diagnostics_for(self, name):
        """返回与指定 Skill 名称关联的全部加载诊断。"""
        return tuple(
            diagnostic
            for diagnostic in self._snapshot.diagnostics
            if diagnostic.skill_name == name
        )

    def format_skills(self):
        """把当前可用 Skill 和诊断格式化为 /skills 文本。"""
        lines = ["Available skills:"]
        if self._snapshot.skills:
            for name in sorted(self._snapshot.skills):
                skill = self._snapshot.skills[name]
                lines.extend(
                    [
                        "",
                        f"/{skill.name}",
                        f"  {skill.description}",
                        f"  Source: {skill.source}",
                    ]
                )
        else:
            lines.extend(
                [
                    "",
                    "No skills are available.",
                    f"Add custom skills under: {self.custom_dir}",
                    "Expected layout: skills/<name>/SKILL.md",
                ]
            )

        if self._snapshot.diagnostics:
            lines.extend(["", "Diagnostics:"])
            for diagnostic in sorted(
                self._snapshot.diagnostics,
                key=lambda item: (
                    item.path,
                    item.severity,
                    item.message,
                ),
            ):
                label = diagnostic.skill_name or Path(diagnostic.path).name
                lines.extend(
                    [
                        "",
                        f"- {diagnostic.source}/{label}: "
                        f"{diagnostic.severity}",
                        f"  {diagnostic.message}",
                        f"  Path: {diagnostic.path}",
                    ]
                )

        return "\n".join(lines)

    @staticmethod
    def _deduplicate_source(skills, source, diagnostics=None):
        """拒绝同一来源的重复名称，并把错误追加到诊断集合。"""
        diagnostics = diagnostics if diagnostics is not None else []
        grouped = {}
        for skill in skills:
            grouped.setdefault(skill.name, []).append(skill)

        result = {}
        for name, candidates in grouped.items():
            if len(candidates) == 1:
                result[name] = candidates[0]
                continue

            for skill in candidates:
                diagnostics.append(
                    SkillDiagnostic(
                        severity="error",
                        source=source,
                        path=skill.skill_file,
                        message=(
                            f'Duplicate Skill name "{name}" '
                            f'in {source} source.'
                        ),
                        skill_name=name,
                    )
                )

        return result
