"""MyAgent 的本地路径规范化工具。"""

from pathlib import Path


def resolve_path(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """展开用户目录，并将路径规范化为基于可选基目录的绝对路径。"""
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_absolute() and base_dir is not None:
        resolved_path = Path(base_dir).expanduser() / resolved_path
    return resolved_path.resolve()
