"""MyAgent 统一命令行入口与子命令分发。"""

import argparse
import json
from collections.abc import Sequence

from channel.web.__main__ import main as run_web_server
from cli.chat import run_chat
from config import conf, drag_sensitive, load_config


def build_parser() -> argparse.ArgumentParser:
    """创建 MyAgent 的统一命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="myagent",
        description="MyAgent 本地 Agent 与 Web Console 启动入口。",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="启动交互式命令行聊天")
    subparsers.add_parser("web", help="启动本机 Web Console")
    subparsers.add_parser("config", help="输出脱敏后的当前有效配置")
    return parser


def show_config(*, load_config_func=None, config_func=None, output_func=None) -> int:
    """加载并以 JSON 输出脱敏后的有效运行配置。"""
    load_config_func = load_config_func or load_config
    config_func = config_func or conf
    output_func = output_func or print
    try:
        load_config_func()
    except Exception as exc:
        output_func(f"MyAgent 配置加载失败：{exc}")
        return 1
    safe_values = drag_sensitive(dict(config_func()))
    output_func(json.dumps(safe_values, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """解析子命令并启动聊天、Web 服务或配置查看流程。"""
    args = build_parser().parse_args(argv)
    if args.command in {None, "chat"}:
        return run_chat()
    if args.command == "web":
        return run_web_server()
    if args.command == "config":
        return show_config()
    raise RuntimeError(f"未知命令：{args.command}")
