# encoding:utf-8
"""解析 SKILL.md 的最小 frontmatter。"""

import re

from agent.skills.types import ParsedFrontmatter


_FIELD_RE = re.compile(r"^([A-Za-z0-9_-]+):[ \t]*(.*)$")
_REQUIRED_FIELDS = {"name", "description"}
_QUOTE_CHARS = {'"', "'"}


class FrontmatterError(ValueError):
    """表示 SKILL.md 的必填 frontmatter 无法安全解析。"""


def parse_frontmatter(content: str) -> ParsedFrontmatter:
    """解析最小 frontmatter，并返回必填元数据、正文和警告。"""
    if content.startswith("﻿"):
        content = content[1:]

    lines = content.splitlines()
    if not lines or lines[0].rstrip() != "---":
        raise FrontmatterError("SKILL.md must start with a standalone --- line")

    closing_index = next(
        (
            index
            for index in range(1, len(lines))
            if lines[index].rstrip() == "---"
        ),
        None,
    )
    if closing_index is None:
        raise FrontmatterError(
            "SKILL.md frontmatter is missing its closing --- line"
        )

    fields = {}
    warnings = []
    for raw_line in lines[1:closing_index]:
        if not raw_line.strip():
            continue

        match = _FIELD_RE.fullmatch(raw_line)
        if not match:
            warnings.append(
                f'Unsupported frontmatter syntax "{raw_line.strip()}"; '
                "MyAgent ignored it."
            )
            continue

        key, value = match.groups()
        value = value.strip()
        if key in fields:
            if key in _REQUIRED_FIELDS:
                raise FrontmatterError(f"Duplicate field: {key}")
            warnings.append(
                f'Duplicate unsupported field "{key}"; MyAgent ignored it.'
            )
            continue

        fields[key] = value
        if key not in _REQUIRED_FIELDS:
            warnings.append(
                f'Unsupported frontmatter field "{key}"; '
                "MyAgent will not enforce it."
            )

    for key in ("name", "description"):
        value = fields.get(key, "")
        if not value:
            raise FrontmatterError(f"Missing or empty required field: {key}")
        if value[0] in _QUOTE_CHARS or value[-1] in _QUOTE_CHARS:
            raise FrontmatterError(
                f"Required field {key} must be an unquoted single-line value"
            )

    body = "\n".join(lines[closing_index + 1:]).strip()
    return ParsedFrontmatter(
        name=fields["name"],
        description=fields["description"],
        body=body,
        warnings=tuple(warnings),
    )
