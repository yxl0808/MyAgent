import tempfile
import unittest
from pathlib import Path

from agent.skills.frontmatter import FrontmatterError, parse_frontmatter
from agent.skills.loader import SkillLoader
from agent.skills.manager import SkillManager


class FrontmatterTest(unittest.TestCase):
    """验证最小 frontmatter 的解析规则。"""

    def test_parse_minimal_frontmatter_and_body(self):
        """验证必填字段和 Markdown 正文能被准确提取。"""
        parsed = parse_frontmatter(
            "---\n"
            "name: systematic-debugging\n"
            "description: Investigate failures systematically\n"
            "---\n"
            "# Workflow\n\nFind the root cause.\n"
        )

        self.assertEqual(parsed.name, "systematic-debugging")
        self.assertEqual(
            parsed.description,
            "Investigate failures systematically",
        )
        self.assertEqual(parsed.body, "# Workflow\n\nFind the root cause.")
        self.assertEqual(parsed.warnings, ())

    def test_unknown_fields_warn_without_becoming_active_metadata(self):
        """验证未知字段只告警且不会获得执行语义。"""
        parsed = parse_frontmatter(
            "---\n"
            "name: code-review\n"
            "description: Review code\n"
            "allowed-tools: read,bash\n"
            "---\n"
            "Review carefully.\n"
        )

        self.assertEqual(
            parsed.warnings,
            (
                'Unsupported frontmatter field "allowed-tools"; '
                "MyAgent will not enforce it.",
            ),
        )

    def test_missing_or_quoted_required_fields_are_rejected(self):
        """验证缺少结构或使用引号的必填字段会被拒绝。"""
        invalid_documents = (
            "description: Missing delimiters\n",
            "---\ndescription: Missing name\n---\nBody\n",
            '---\nname: "quoted-name"\ndescription: Bad name\n---\nBody\n',
            '---\nname: valid-name\ndescription: "quoted description"\n---\nBody\n',
        )

        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(FrontmatterError):
                    parse_frontmatter(document)

    def test_duplicate_required_field_is_rejected(self):
        """验证重复必填字段不会被静默覆盖。"""
        with self.assertRaisesRegex(FrontmatterError, "Duplicate field: name"):
            parse_frontmatter(
                "---\n"
                "name: first\n"
                "name: second\n"
                "description: Duplicate\n"
                "---\n"
                "Body\n"
            )


class SkillLoaderTest(unittest.TestCase):
    """验证单来源扫描、错误隔离和格式告警。"""

    def _write_skill(self, root, directory, name=None, extra=""):
        """在临时来源中写入一个可定制的 Skill 测试夹具。"""
        skill_dir = Path(root) / directory
        skill_dir.mkdir(parents=True)
        declared_name = name or directory
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {declared_name}\n"
            f"description: Description for {declared_name}\n"
            f"{extra}"
            "---\n"
            f"# {declared_name}\n",
            encoding="utf-8",
        )
        return skill_dir

    def test_load_valid_skill_and_keep_absolute_paths(self):
        """验证合法 Skill 被加载且路径转换为绝对路径。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = self._write_skill(temp_dir, "code-review")
            result = SkillLoader(Path(temp_dir), "custom").load()

        self.assertEqual([skill.name for skill in result.skills], ["code-review"])
        skill = result.skills[0]
        self.assertEqual(skill.source, "custom")
        self.assertEqual(skill.skill_dir, str(skill_dir.resolve()))
        self.assertEqual(
            skill.skill_file,
            str((skill_dir / "SKILL.md").resolve()),
        )
        self.assertEqual(result.diagnostics, ())

    def test_invalid_skill_does_not_block_valid_sibling(self):
        """验证一个损坏 Skill 不会阻止同来源的合法兄弟项。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_skill(temp_dir, "valid-skill")
            self._write_skill(
                temp_dir,
                "wrong-directory",
                name="different-name",
            )
            result = SkillLoader(Path(temp_dir), "custom").load()

        self.assertEqual([skill.name for skill in result.skills], ["valid-skill"])
        self.assertEqual(len(result.diagnostics), 1)
        self.assertEqual(result.diagnostics[0].severity, "error")
        self.assertIn(
            "does not match directory",
            result.diagnostics[0].message,
        )

    def test_unknown_field_warns_but_skill_loads(self):
        """验证未知字段产生警告但不阻止 Skill 加载。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_skill(
                temp_dir,
                "external-skill",
                extra="allowed-tools: read,bash\n",
            )
            result = SkillLoader(Path(temp_dir), "custom").load()

        self.assertEqual(
            [skill.name for skill in result.skills],
            ["external-skill"],
        )
        self.assertEqual(result.diagnostics[0].severity, "warning")
        self.assertIn("allowed-tools", result.diagnostics[0].message)

    def test_only_real_visible_first_level_directories_are_scanned(self):
        """验证加载器只扫描可见的一级真实目录。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_skill(root, "visible-skill")
            self._write_skill(root, ".hidden-skill")
            (root / "README.md").write_text("not a skill", encoding="utf-8")
            nested = root / "container" / "nested-skill"
            nested.mkdir(parents=True)
            (nested / "SKILL.md").write_text(
                "---\n"
                "name: nested-skill\n"
                "description: nested\n"
                "---\n"
                "Body",
                encoding="utf-8",
            )
            result = SkillLoader(root, "builtin").load()

        self.assertEqual(
            [skill.name for skill in result.skills],
            ["visible-skill"],
        )
        self.assertTrue(
            any(
                "SKILL.md" in diagnostic.message
                for diagnostic in result.diagnostics
            )
        )


class SkillManagerTest(unittest.TestCase):
    """验证两个 Skill 来源的合并、刷新和展示。"""

    def _write_skill(self, root, name, description=None):
        """在指定来源中写入一个最小 Skill。"""
        skill_dir = Path(root) / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            f"name: {name}\n"
            f"description: {description or name}\n"
            "---\n"
            f"# {name}\n",
            encoding="utf-8",
        )
        return skill_dir

    def test_custom_skill_cannot_override_builtin(self):
        """验证同名自定义 Skill 被拒绝且内置版本保留。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            builtin = root / "builtin"
            custom = root / "custom"
            self._write_skill(builtin, "code-review", "Builtin review")
            self._write_skill(custom, "code-review", "Custom review")
            manager = SkillManager(builtin, custom)

            snapshot = manager.refresh()

        self.assertEqual(
            snapshot.skills["code-review"].description,
            "Builtin review",
        )
        self.assertTrue(
            any(
                diagnostic.skill_name == "code-review"
                and diagnostic.severity == "error"
                and "conflicts with builtin" in diagnostic.message
                for diagnostic in snapshot.diagnostics
            )
        )

    def test_refresh_reflects_add_modify_and_delete(self):
        """验证刷新能反映 Skill 的新增、修改和删除。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            builtin = root / "builtin"
            custom = root / "custom"
            custom.mkdir()
            manager = SkillManager(builtin, custom)
            self.assertEqual(dict(manager.refresh().skills), {})

            skill_dir = self._write_skill(
                custom,
                "dynamic-skill",
                "Version one",
            )
            self.assertEqual(
                manager.refresh().skills["dynamic-skill"].description,
                "Version one",
            )

            self._write_skill(
                custom,
                "dynamic-skill",
                "Version two",
            )
            self.assertEqual(
                manager.refresh().skills["dynamic-skill"].description,
                "Version two",
            )

            (skill_dir / "SKILL.md").unlink()
            self.assertNotIn("dynamic-skill", manager.refresh().skills)

    def test_format_skills_is_stable_and_includes_diagnostics(self):
        """验证列表排序稳定并同时展示来源和诊断。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            builtin = root / "builtin"
            custom = root / "custom"
            self._write_skill(builtin, "zeta-skill", "Zeta")
            self._write_skill(custom, "alpha-skill", "Alpha")
            bad_dir = custom / "broken-skill"
            bad_dir.mkdir(parents=True)
            manager = SkillManager(builtin, custom)
            manager.refresh()

            rendered = manager.format_skills()

        self.assertLess(
            rendered.index("/alpha-skill"),
            rendered.index("/zeta-skill"),
        )
        self.assertIn("Source: custom", rendered)
        self.assertIn("Source: builtin", rendered)
        self.assertIn("Diagnostics:", rendered)
        self.assertIn("broken-skill", rendered)


if __name__ == "__main__":
    unittest.main()
