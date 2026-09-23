# encoding:utf-8
"""MyAgent 兼容的命令行聊天启动入口。"""

from agent import create_agent
from cli.chat import resolve_agent_result
from cli.chat import run_chat
from config import load_config


_resolve_agent_result = resolve_agent_result


def main() -> int:
    """保留 `python main.py` 的交互式聊天兼容入口。"""
    return run_chat(
        load_config_func=load_config,
        create_agent_func=create_agent,
    )


if __name__ == "__main__":
    raise SystemExit(main())
