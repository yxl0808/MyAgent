"""MyAgent 交互式命令行聊天实现。"""

from types import GeneratorType

from agent import create_agent
from common.console import print_safely
from config import load_config


def resolve_agent_result(value):
    """提取非流式 Agent.run() 返回的结果对象。"""
    if not isinstance(value, GeneratorType):
        return value

    try:
        next(value)
    except StopIteration as stop:
        return stop.value

    raise RuntimeError("非流式 Agent.run() 意外产生了事件")


def run_chat(
    *,
    load_config_func=None,
    create_agent_func=None,
    input_func=None,
    output_func=None,
) -> int:
    """加载配置后启动兼容旧入口的交互式 Agent 聊天循环。"""
    load_config_func = load_config_func or load_config
    create_agent_func = create_agent_func or create_agent
    input_func = input_func or input
    output_func = output_func or print_safely
    try:
        load_config_func()
        agent = create_agent_func()
    except Exception as exc:
        output_func(f"MyAgent 启动失败：{exc}")
        return 1

    output_func("MyAgent 已启动，输入 exit 或 quit 退出。")
    while True:
        try:
            user_message = input_func("> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_func("")
            break
        if user_message.lower() in {"exit", "quit"}:
            break
        if not user_message:
            continue

        result = resolve_agent_result(agent.run(user_message))
        output_func(result.final_answer)
    return 0
