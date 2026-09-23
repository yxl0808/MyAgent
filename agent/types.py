# encoding:utf-8
"""
MyAgent Agent 核心 — 数据类型定义
=================================
定义 Agent 运行过程中涉及的所有数据结构。

三个核心概念:
  1. AgentActionType: Agent 一轮循环能做什么（三种可能）
  2. ToolResult:      工具执行完后返回的结果
  3. AgentAction:     一轮循环的完整记录（谁做了什么 + 结果）
  4. AgentResult:     一次完整对话的最终结果
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AgentActionType(Enum):
    """
    Agent 在一轮推理中可能产生的动作类型。

    模型的一条回复可能是以下三种之一:

      TOOL_USE:    "我需要调用工具来完成任务"
                   回复中包含 tool_calls，Agent 需要执行工具并继续循环

      THINKING:    "让我分析一下当前情况..."
                   模型的内部推理过程（DeepSeek thinking mode / Claude extended thinking）
                   这不是最终答案，Agent 继续等待或者执行工具

      FINAL_ANSWER: "根据查询结果，答案是..."
                   模型认为任务已完成，这是给用户的最终回复
                   Agent 循环在此终止
    """
    TOOL_USE = "tool_use"        # 模型请求调用工具
    THINKING = "thinking"         # 模型内部推理（可选，不影响循环逻辑）
    FINAL_ANSWER = "final_answer"  # 模型给出最终答案，循环结束


@dataclass
class ToolResult:
    """
    工具执行的一次完整记录。

    每当 LLM 请求调用一个工具（如 bash、web_search），Agent 执行该工具后，
    用 ToolResult 记录：调了什么工具、传了什么参数、返回了什么、花了几秒。

    字段说明:
        tool_name:      工具名，如 "bash" / "web_search" / "read"
        input_params:   传给工具的参数，如 {"command": "ls -la"}
        output:         工具返回的内容，类型取决于工具（通常是 str 或 dict）
                        - bash 返回 stdout 字符串
                        - web_search 返回搜索结果列表
        success:        工具是否成功执行（True 表示没有异常）
        error:          如果失败，这里是错误信息字符串；成功则为 None
        duration_ms:    工具执行耗时（毫秒），用于日志统计
    """
    tool_name: str                    # 工具名称
    input_params: Dict[str, Any]      # 调用时传入的参数
    output: Any = ""                  # 工具返回的内容（字符串或其他类型）
    success: bool = True              # 是否执行成功
    error: Optional[str] = None       # 失败时的错误信息
    duration_ms: float = 0.0          # 执行耗时（毫秒）


@dataclass
class AgentAction:
    """
    Agent 一轮循环的完整记录。

    Agent 每次调用 LLM 后，LLM 的返回都产生一个 AgentAction。
    一轮循环的流程:
      1. LLM 返回 → 创建 AgentAction，记录模型说的话和类型
      2. 如果是 TOOL_USE → 执行工具 → 把结果填入 tool_result
      3. 如果是 FINAL_ANSWER → 循环终止，返回给用户

    字段说明:
        id:           本轮操作的唯一 ID（自动生成 UUID）
        action_type:  模型这次想干嘛（TOOL_USE / THINKING / FINAL_ANSWER）
        content:      模型的文字输出。TOOL_USE 时是"让我查一下..."之类的解释，
                      FINAL_ANSWER 时是给用户的最终回复
        thinking:     模型内部推理内容（DeepSeek thinking mode / Claude extended thinking）
                      没有则为空字符串
        tool_calls:   模型请求的工具调用列表（TOOL_USE 时非空）
                      格式: [{"id":"call_xxx","name":"bash","arguments":{"command":"ls"}}, ...]
        tool_result:  工具执行完后填入（TOOL_USE + 执行完成后非空）
        usage:        本轮 LLM 调用的 token 消耗 {"input": N, "output": N}
        step:         这是第几步（从 1 开始计数），用于日志和调试
        timestamp:    创建时间（Unix 时间戳）
    """
    action_type: AgentActionType          # 本轮动作类型
    content: str = ""                      # 模型的文字输出
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 自动生成唯一 ID
    thinking: str = ""                     # 内部推理内容（可选）
    tool_calls: list = field(default_factory=list)       # 工具调用列表
    tool_result: Optional[ToolResult] = None             # 工具执行结果
    usage: Dict[str, int] = field(default_factory=dict)  # token 用量
    step: int = 0                          # 当前步数
    timestamp: float = field(default_factory=time.time)  # 创建时间


@dataclass
class AgentResult:
    """
    一次完整对话的最终结果。

    当 Agent 循环结束时（无论是正常结束还是异常中断），
    将所有信息打包为 AgentResult 返回给调用方（Channel/Bridge 层）。

    字段说明:
        final_answer:  给用户的最终回复文本。
                       正常结束: LLM 的最后一条 FINAL_ANSWER 内容
                       异常中断: 错误描述信息
        step_count:    总共执行了多少步（调了多少次 LLM）。
                       正常情况 1-5 步，复杂任务可能 10+ 步
        actions:       每一步的详细记录（AgentAction 列表）。
                       记录了"谁在什么时候做了什么"，用于日志和调试
        success:       是否成功完成。
                       True:  LLM 正常给出最终答案
                       False: 达到最大步数上限 / LLM 调用失败 / 其他异常
        error:         失败时的错误信息，成功时为 None
        total_usage:   整次对话的总 token 消耗 {"input": N, "output": N}
    """
    final_answer: str = ""                           # 最终回复文本
    step_count: int = 0                              # 总步数
    actions: list = field(default_factory=list)      # 每一步的 AgentAction 记录
    success: bool = True                             # 是否成功
    error: Optional[str] = None                      # 失败时的错误信息
    total_usage: Dict[str, int] = field(default_factory=dict)  # 总 token 用量
