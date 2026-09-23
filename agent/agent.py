# encoding:utf-8
"""
MyAgent Agent 核心 — Agent 类
=============================
Agent 是整个项目的"大脑"，负责：

  1. 接收用户消息
  2. 构建系统提示词（注入工具列表、技能、记忆、工作区信息）
  3. 调用 LLM（通过 models/ 适配层）
  4. 解析 LLM 返回（text / tool_calls / thinking）
  5. 如果是工具调用 → 找到对应工具 → 执行 → 把结果发回 LLM → 回到步骤 3
  6. 如果是最终答案 → 返回给用户 → 结束

这是 Agent Harness 的核心循环：思考 → 行动 → 观察 → 再思考 → ... → 回答

调用方式:
    from models import create_model
    from agent import Agent

    llm = create_model("openai")
    agent = Agent(
        model=llm,
        system_prompt="你是一个有用的助手",
        tools=[bash_tool, read_tool, web_search_tool],
        max_steps=20,
    )
    result = agent.run("帮我查一下今天的天气")
    print(result.final_answer)  # "北京今天 35°C，注意防暑"
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.types import AgentAction, AgentActionType, AgentResult, ToolResult
from agent.tools.base import BaseTool  # 后面会创建

logger = logging.getLogger(__name__)

_TOOL_FAILURE_PREFIXES = ("error:", "err:", "[exit code ")
_MAX_CONSECUTIVE_TOOL_FAILURES = 3
_MAX_CONSECUTIVE_DUPLICATE_TOOL_CALLS = 3
_LIGHTWEIGHT_MODE_BLOCKERS = ("http://", "https://", "www.", "最新", "今天", "昨天", "明天", "当前", "目前", "现在", "搜索", "查询", "联网", "网页", "官网", "文件", "目录", "工作区", "代码", "命令", "终端", "运行", "执行", "读取", "打开", "写入", "创建", "修改", "编辑", "删除", "下载", "bash", "python", "powershell", ".py", ".md", ".json")


def _cancel_requested(cancel_event) -> bool:
    """返回可选线程安全取消信号是否已经触发。"""
    return cancel_event is not None and cancel_event.is_set()


class Agent:
    """
    AI Agent — 具备工具调用能力的智能体。

    Agent 本身不调用任何 LLM API（那是 models/ 的事），
    也不执行任何工具（那是 tools/ 的事）。
    Agent 只做一件事：协调 LLM 和工具之间的对话循环。

    属性:
        model:          BaseLLM 实例（通过 models/create_model 创建）
        system_prompt:  系统提示词字符串（定义 Agent 的角色和行为）
        tools:          工具列表（BaseTool 子类实例）
        max_steps:      最大工具调用步数（防止死循环）
        messages:       对话历史 [{"role":"user","content":"..."}, ...]
        actions:        本轮对话的每一步记录 [AgentAction, ...]
    """

    TOOL_OUTPUT_MAX_CHARS = 12000

    def __init__(self, model, system_prompt: str = "",
                 tools: Optional[List[BaseTool]] = None,
                 max_steps: int = 20,
                 workspace_dir: str = "",
                 memory_manager=None,
                 skill_manager=None):
        """
        初始化 Agent 实例。

        入参:
            model:          BaseLLM 实例，由 models/create_model() 创建。
                            这是 Agent 唯一依赖的外部组件——只需要一个能 chat() 的对象。
            system_prompt:  系统提示词，告诉 Agent"你是谁、你能做什么"。
                            例如: "你是一个 Python 编程助手，可以使用 bash 执行代码"
            tools:          工具列表，Agent 可以调用的所有工具。
                            每个工具都是 BaseTool 子类实例（有 name/description/execute）
            max_steps:      最大工具调用步数上限。
                            每次 Agent 调 LLM 算一步，工具执行不算。
                            设上限是为了防止 LLM 陷入"调工具→不满意→再调→还不满意"的死循环
            workspace_dir:  Agent 的工作区目录。
                            存放技能、记忆文件、临时文件等。
                            例如: "~/myagent" 或 "/home/user/myagent"
            memory_manager: MemoryManager 实例（可选）。
                            负责长期记忆的读写检索。为 None 时内存功能不启用
            skill_manager:  SkillManager 实例（可选）。
                            负责技能（Skills）的加载和执行。为 None 时技能功能不启用
        """
        # ── 核心依赖 ──
        self.model = model                    # LLM 适配器（唯一的外部依赖）
        self.system_prompt = system_prompt     # 系统提示词（基础版本，运行时会扩展）
        self.max_steps = max_steps            # 最大步数上限

        # ── 工具 ──
        self.tools: List[BaseTool] = []
        if tools:
            for tool in tools:
                self.add_tool(tool)

        # ── 对话历史 ──
        # messages 在多次 agent.run() 调用之间保持，实现"多轮对话"能力
        # 格式: [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]
        # 调用 agent.clear_history() 可以清空
        self.messages: List[Dict[str, Any]] = []

        # ── 本轮运行记录 ──
        # actions 在每次 agent.run() 开始时清空，记录本轮每一步的详情
        self.actions: List[AgentAction] = []

        # ── 可选组件 ──
        self.workspace_dir = workspace_dir      # 工作区路径
        self.memory_manager = memory_manager    # 记忆管理器（可为 None）
        self.skill_manager = skill_manager      # 技能管理器（可为 None）

        # ── 上下文管理 ──
        # 从 config 读取上下文大小限制，未配置则使用默认值
        from config import conf
        self.max_context_tokens = conf().get("agent_max_context_tokens", 50000)
        self.max_context_turns = conf().get("agent_max_context_turns", 20)
        self.lightweight_mode_enabled = conf().get("agent_lightweight_mode_enabled", True)
        self.lightweight_query_max_chars = conf().get("agent_lightweight_query_max_chars", 180)

    # ── 工具管理 ──────────────────────────────────────────────

    def add_tool(self, tool: BaseTool):
        """
        注册一个工具到 Agent。

        工具注册后，Agent 在每次构建系统提示词时，
        会将所有已注册工具的名称、描述和参数格式注入 prompt，
        LLM 看到后就知道"我能调用哪些工具"。

        入参:
            tool: BaseTool 子类实例（有 name / description / parameters / execute 四个要素）
        """
        self.tools.append(tool)
        tool._agent = self
        logger.debug("[Agent] Tool registered: %s", tool.name)

    def _find_tool(self, name: str) -> Optional[BaseTool]:
        """
        根据工具名查找对应的工具实例。

        LLM 返回的 tool_calls 里只有工具名（字符串），Agent 需要在注册的工具列表
        中找到对应的 BaseTool 实例才能调用 .execute()。

        查找逻辑: 线性遍历 self.tools，匹配 name 属性。
        因为工具数量通常 < 20 个，线性查找足够快，不需要用字典优化。

        入参:
            name: 工具名，如 "bash" / "web_search"

        返回:
            找到的 BaseTool 实例，找不到则返回 None
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        # 找不到：可能是 LLM 幻觉出了一个不存在的工具名
        logger.warning("[Agent] Tool not found: %s (available: %s)",
                       name, [t.name for t in self.tools])
        return None

    @classmethod
    def _prepare_tool_output(cls, output: Any) -> str:
        """将工具结果序列化并压缩为适合模型上下文的文本。"""
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                text = str(output)
        if len(text) <= cls.TOOL_OUTPUT_MAX_CHARS:
            return text

        marker = (
            "\n\n[tool output truncated; "
            f"original length={len(text)} chars]\n\n"
        )
        available = max(1, cls.TOOL_OUTPUT_MAX_CHARS - len(marker))
        head_length = available // 2
        tail_length = available - head_length
        return text[:head_length] + marker + text[-tail_length:]

    def _build_max_steps_result(
        self,
        final_text: str,
        step_count: int,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> AgentResult:
        """在步数上限处汇总模型文字和工具结果后返回失败结果。"""
        progress: list[str] = []
        for action in self.actions:
            if action.content.strip():
                progress.append(f"步骤 {action.step}：{action.content.strip()}")
            if action.tool_result is not None:
                tool_output = self._prepare_tool_output(action.tool_result.output)
                if tool_output:
                    progress.append(
                        f"步骤 {action.step}：工具 {action.tool_result.tool_name} 返回：\n"
                        f"{tool_output}"
                    )
        progress_text = "\n\n".join(progress)
        if progress_text:
            answer = (
                f"任务达到最大执行步数（{self.max_steps} 步），已停止继续调用工具。\n\n"
                "以下是目前已经获得的阶段性结果：\n"
                f"{progress_text}"
            )
        else:
            answer = final_text or (
                f"任务达到最大执行步数（{self.max_steps} 步），"
                "暂时没有可汇总的阶段性结果。"
            )
        answer = self._prepare_tool_output(answer)
        return AgentResult(
            final_answer=answer,
            step_count=step_count,
            actions=self.actions,
            success=False,
            error="max_steps_exceeded",
            total_usage={"input": total_input_tokens, "output": total_output_tokens},
        )

    @staticmethod
    def _tool_call_signature(tool_name: str, tool_args: Any):
        """生成不包含调用 ID 的规范化工具调用指纹。"""
        normalized_args = json.dumps(
            tool_args,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return tool_name, normalized_args

    def _build_duplicate_tool_call_result(
        self,
        tool_name: str,
        step_count: int,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> AgentResult:
        """汇总已有进度并返回重复工具调用失败结果。"""
        progress: list[str] = []
        for action in self.actions:
            if action.content.strip():
                progress.append(f"步骤 {action.step}：{action.content.strip()}")
            if action.tool_result is not None:
                tool_output = self._prepare_tool_output(action.tool_result.output)
                if tool_output:
                    progress.append(
                        f"步骤 {action.step}：工具 {action.tool_result.tool_name} 返回：\n"
                        f"{tool_output}"
                    )
        progress_text = "\n\n".join(progress)
        answer = (
            f"检测到工具 {tool_name} 连续使用完全相同的参数，"
            "已停止继续调用。"
        )
        if progress_text:
            answer += "\n\n以下是目前已经获得的阶段性结果：\n" + progress_text
        return AgentResult(
            final_answer=self._prepare_tool_output(answer),
            step_count=step_count,
            actions=self.actions,
            success=False,
            error="duplicate_tool_call",
            total_usage={"input": total_input_tokens, "output": total_output_tokens},
        )

    def _append_skipped_tool_messages(
        self,
        tool_calls: list[dict[str, Any]],
        start_index: int,
        reason: str,
    ) -> None:
        """为提前停止时尚未执行的工具调用补齐响应消息。"""
        for pending_call in tool_calls[start_index:]:
            self.messages.append({
                "role": "tool",
                "tool_call_id": pending_call.get("id", ""),
                "content": f"Error: tool call skipped because {reason}.",
            })

    @staticmethod
    def _tool_output_indicates_failure(output: Any) -> bool:
        """根据工具统一错误前缀识别未抛异常的执行失败。"""
        if not isinstance(output, str):
            return False
        return output.lstrip().lower().startswith(_TOOL_FAILURE_PREFIXES)

    # ── Token 估算（上下文管理用） ─────────────────────────────
    # 不调 API 就无法精确知道 token 数（每个模型的 tokenizer 不同），
    # 但用字符数估算已经足够给上下文截断做参考——不需要精确到个位数。

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        """
        根据字符类型估算一段文字的 token 数量。

        估算规则（经验值，不是精确计算）:
          - 中文字符（Unicode > 127）: 约 1.5 token/字
            例: "你好" → 约 3 tokens
          - 英文字符（ASCII）:       约 0.25 token/字母（4 个字母 ≈ 1 token）
            例: "hello" → 约 1.25 tokens
          - 空字符串 → 0

        为什么不用 tiktoken 库精确计算:
          精确计算需要导入每个模型的 tokenizer（不同模型 tokenizer 不同），
          增加依赖和复杂度。估算的误差在 ±20% 以内，对上下文截断来说够用——
          因为我们只判断"是否超出上限"，不是"精准到个位数的计数"。

        入参:
            text: 待估算的文本

        返回:
            估算 token 数（整数）
        """
        if not text:
            return 0
        # 统计非 ASCII 字符（中文、日文、韩文、emoji 等）
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ascii_count = len(text) - non_ascii
        # 加权计算: 中文约 1.5 token/字, 英文约 0.25 token/字母
        return int(non_ascii * 1.5 + ascii_count * 0.25) + 1

    # ── 上下文管理 ───────────────────────────────────────────

    def _trim_and_flush(self):
        """
        当对话历史超过最大轮次限制时，把旧对话写入长期记忆并裁剪上下文。
        """
        if self.max_context_turns <= 0:
            return

        user_turns = sum(1 for m in self.messages if m.get("role") == "user")
        if user_turns <= self.max_context_turns:
            return

        discard_user_turns = user_turns - self.max_context_turns
        seen_user_turns = 0
        cutoff_idx = 0

        for i, message in enumerate(self.messages):
            if message.get("role") != "user":
                continue

            seen_user_turns += 1
            if seen_user_turns > discard_user_turns:
                cutoff_idx = i
                break

        if cutoff_idx <= 1:
            return

        discarded = self.messages[1:cutoff_idx]
        if not discarded:
            return

        if self.memory_manager:
            self.memory_manager.flush(discarded, reason="trim")

        self.messages = [self.messages[0]] + self.messages[cutoff_idx:]
        logger.info(
            "[Agent] Trimmed %d messages after %d user turns",
            len(discarded),
            discard_user_turns,
        )

    def _overflow_flush(self):
        """
        当对话历史超过 token 上限时，把旧消息写入长期记忆并裁剪上下文。
        """
        if self.max_context_tokens <= 0 or len(self.messages) <= 1:
            return

        def message_tokens(message: Dict[str, Any]) -> int:
            content = message.get("content", "")
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            return self._estimate_text_tokens(text)

        total_tokens = sum(message_tokens(message) for message in self.messages)
        if total_tokens <= self.max_context_tokens:
            return

        target_tokens = max(1, int(self.max_context_tokens * 0.8))
        system_messages = (
            [self.messages[0]]
            if self.messages and self.messages[0].get("role") == "system"
            else []
        )
        body_messages = self.messages[len(system_messages):]
        kept_reversed = []
        kept_tokens = sum(message_tokens(message) for message in system_messages)

        for message in reversed(body_messages):
            token_count = message_tokens(message)
            if kept_tokens + token_count <= target_tokens or not kept_reversed:
                kept_reversed.append(message)
                kept_tokens += token_count
                continue
            break

        kept_messages = list(reversed(kept_reversed))
        discarded_count = len(body_messages) - len(kept_messages)
        if discarded_count <= 0:
            return

        discarded = body_messages[:discarded_count]
        if self.memory_manager:
            self.memory_manager.flush(discarded, reason="overflow")

        self.messages = system_messages + kept_messages
        logger.info(
            "[Agent] Overflow-trimmed %d messages (%d -> %d estimated tokens)",
            len(discarded),
            total_tokens,
            kept_tokens,
        )

    def _threshold_flush(self):
        """
        当对话历史接近 token 上限时，提前把旧消息写入长期记忆。
        """
        if (
            self.max_context_tokens <= 0
            or len(self.messages) <= 1
            or not self.memory_manager
        ):
            return

        def message_tokens(message: Dict[str, Any]) -> int:
            content = message.get("content", "")
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            return self._estimate_text_tokens(text)

        total_tokens = sum(message_tokens(message) for message in self.messages)
        if total_tokens >= self.max_context_tokens:
            return

        threshold_tokens = int(self.max_context_tokens * 0.8)
        if total_tokens < threshold_tokens:
            return

        flushed_count = getattr(self, "_threshold_flushed_message_count", 0)
        if flushed_count >= len(self.messages):
            return

        system_count = (
            1
            if self.messages and self.messages[0].get("role") == "system"
            else 0
        )
        body_messages = self.messages[system_count:]
        kept_reversed = []
        kept_tokens = sum(
            message_tokens(message)
            for message in self.messages[:system_count]
        )

        for message in reversed(body_messages):
            token_count = message_tokens(message)
            if kept_tokens + token_count <= threshold_tokens or not kept_reversed:
                kept_reversed.append(message)
                kept_tokens += token_count
                continue
            break

        kept_count = len(kept_reversed)
        flush_count = len(body_messages) - kept_count
        if flush_count <= 0:
            return

        messages_to_flush = body_messages[:flush_count]
        ok = self.memory_manager.flush(messages_to_flush, reason="threshold")
        if ok is not False:
            self._threshold_flushed_message_count = len(self.messages)

        logger.info(
            "[Agent] Threshold-flushed %d messages at %d estimated tokens",
            len(messages_to_flush),
            total_tokens,
        )

    # ── 系统提示词构建 ───────────────────────────────────────
    # 每次调用 LLM 前都重新构建（因为工具/技能/记忆可能已变化），
    # 构建好的 prompt 作为 messages[0] 的 system 消息发给 LLM。

    def _is_lightweight_question(self, query: str, active_skill=None) -> bool:
        """判断当前问题是否适合跳过工具和记忆检索。"""
        normalized_query = query.strip().lower()
        if (
            not self.lightweight_mode_enabled
            or active_skill is not None
            or not normalized_query
            or len(normalized_query) > self.lightweight_query_max_chars
        ):
            return False
        return not any(
            blocker in normalized_query
            for blocker in _LIGHTWEIGHT_MODE_BLOCKERS
        )

    def _build_system_prompt(self, query: str = "", active_skill=None,
                             include_tools: bool = True,
                             include_memory: bool = True) -> str:
        """
        构建完整的系统提示词，包含 Agent 需要知道的全部上下文。

        构建顺序:
          1. 基础 system_prompt（告诉 LLM "你是谁"）
          2. 当前时间（让 LLM 知道"现在是什么时候"）
          3. 工作区路径（让 LLM 知道在哪个目录下操作）
          4. 工具列表（让 LLM 知道可以用哪些工具 + 每个工具的参数格式）
          5. 当前显式激活的 Skill（仅本轮存在）
          6. 记忆检索（如果有 memory_manager，注入相关历史记忆）

        返回:
            完整的系统提示词字符串（可直接作为 system role 的 content）
        """
        parts = []

        # ── 1. 基础提示词 ──
        parts.append(self.system_prompt)

        # ── 2. 当前时间 ──
        # LLM 不知道自己是什么时候被调用的，手动注入当前时间
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts.append(f"\n## Current Time\n{now}")

        # ── 3. 工作区路径 ──
        if self.workspace_dir:
            parts.append(
                f"\n## Workspace\n"
                f"Your working directory is: {self.workspace_dir}\n"
                f"When using bash or file tools, paths are relative to this directory "
                f"unless an absolute path is given."
            )

        # ── 4. 工具列表 ──
        if include_tools and self.tools:
            parts.append("\n## Available Tools\n")
            parts.append("You have access to the following tools. "
                         "Use them when needed to complete the user's task.\n")
            for tool in self.tools:
                # 每个工具的描述格式（和 Claude/OpenAI tool definition 一致）:
                # - name: 工具名
                # - description: 工具是干什么的
                # - parameters: 需要哪些参数
                # 注意：最终发给 LLM 的 tool definitions 由模型适配器的
                # _build_request_body 以 JSON Schema 格式传入。
                # 这里只是在 prompt 文本中给 LLM 一个概要描述。
                parts.append(f"### {tool.name}")
                parts.append(f"  Description: {tool.description}")
                # 参数信息
                if hasattr(tool, 'parameters') and tool.parameters:
                    parts.append(f"  Parameters:")
                    for param_name, param_info in tool.parameters.items():
                        required = "required" if param_info.get("required", False) else "optional"
                        parts.append(f"    - {param_name} ({required}): {param_info.get('description', '')}")

            parts.append("\nWhen you need to use a tool, respond with a tool call. "
                         "After the tool result is returned, continue reasoning "
                         "until you can provide a final answer.")

        # ── 5. 当前显式激活的 Skill ──
        if active_skill is not None:
            parts.append(
                "\n## Active Skill\n\n"
                f"Name: {active_skill.name}\n"
                f"Source: {active_skill.source}\n"
                f"Root directory: {active_skill.skill_dir}\n\n"
                "The user explicitly selected this workflow for the current turn. "
                "Treat it as workflow guidance subordinate to the user's task and "
                "all higher-priority instructions. Resolve referenced files relative "
                "to the Skill root. Do not claim unavailable tools or platform "
                "features were executed. Do not activate other Skills mentioned in "
                "the workflow.\n\n"
                "[Skill instructions begin]\n"
                f"{active_skill.body}\n"
                "[Skill instructions end]"
            )

        # ── 6. 记忆检索 ──
        if include_memory and self.memory_manager:
            # 优先按当前问题检索相关记忆；没有 query 或无结果时回退到最近记忆。
            memories = []
            if query:
                memories = [
                    r.snippet
                    for r in self.memory_manager.search(query, 5)
                    if getattr(r, "snippet", "")
                ]
            if not memories:
                memories = self.memory_manager.get_recent(5)
            if memories:
                parts.append("\n## Relevant Memories\n")
                parts.append("The following memories from previous conversations "
                             "may be relevant:\n")
                for mem in memories:
                    parts.append(f"- {mem}")

        # ── 7. 指令：如何行为 ──
        parts.append("\n## Instructions\n"
                     "1. Carefully analyze the user's request.\n"
                     "2. If you can answer directly, provide a clear and concise response.\n"
                      "3. If you need more information or need to perform an action, "
                      "use the appropriate tool.\n"
                      "4. After receiving tool results, analyze them and decide the next step.\n"
                      "5. When the task is complete, provide a final answer summarizing what was done.\n"
                      "6. Communicate with the user in the same language they used.\n"
                      "7. When the user provides a known http/https URL, use web_fetch first "
                      "to read it. Do not use bash, curl, wget, or PowerShell merely to "
                      "download or parse a regular web page. Use web_search only when the "
                      "user has not provided a URL.")

        return "\n".join(parts)

    # ── 核心循环 ─────────────────────────────────────────────
    # run() 是 Agent 公开的唯一执行入口。
    # 每次用户发消息，调用 agent.run(user_message) 即可。
# 用户: "帮我创建 hello.py"
#   │
#   ▼
# ┌─ run() 开始 ─────────────────┐
# │                                 │
# │  Step 1: LLM 返回 tool_calls=[bash, "touch hello.py"]   │
# │          → 执行 bash → 工具结果 "文件已创建"             │
# │          → 把结果追加到 messages → 继续循环              │
# │                                 │
# │  Step 2: LLM 返回 tool_calls=[write, "print('hi')"]     │
# │          → 执行 write → 工具结果 "写入成功"               │
# │          → 把结果追加到 messages → 继续循环              │
# │                                 │
# │  Step 3: LLM 返回 text="已创建 hello.py，内容为..."      │
# │          → 没有 tool_calls → FINAL_ANSWER               │
# │          → 退出循环                                      │
# │                                 │
# │  返回 AgentResult(final_answer="已创建 hello.py...")     │
# └────────────────┘


    @staticmethod
    def _parse_skill_command(user_message: str):
        """识别 /skills 或 /<name> 命令，并提取裁剪后的任务正文。"""
        candidate = user_message.lstrip()
        if not candidate.startswith("/"):
            return None

        match = re.match(r"^/(\S+)(?:\s+([\s\S]*))?$", candidate)
        if not match:
            return ("unknown", "", "")

        name = match.group(1)
        task = (match.group(2) or "").strip()
        if name == "skills":
            return ("list", name, task)
        return ("invoke", name, task)

    @staticmethod
    def _command_result(answer: str, success: bool, error: str = None):
        """构造未调用模型和工具的零步命令结果。"""
        return AgentResult(
            final_answer=answer,
            step_count=0,
            actions=[],
            success=success,
            error=error,
            total_usage={"input": 0, "output": 0},
        )

    def _build_cancelled_result(
        self,
        partial_text: str,
        step_count: int,
        total_input_tokens: int,
        total_output_tokens: int,
    ) -> AgentResult:
        """构造 Agent 运行被取消时返回的稳定结果。"""
        return AgentResult(
            final_answer=partial_text,
            step_count=step_count,
            actions=self.actions,
            success=False,
            error="cancelled",
            total_usage={
                "input": total_input_tokens,
                "output": total_output_tokens,
            },
        )

    def _resolve_skill_command(self, parsed_command):
        """刷新并解析 Skill 命令，返回激活对象、任务或直接结果。"""
        kind, name, task = parsed_command
        if self.skill_manager is None:
            return None, None, self._command_result(
                "Skill functionality is disabled.",
                False,
                "skills_disabled",
            )

        if kind == "list" and task:
            return None, None, self._command_result(
                "Usage: /skills",
                False,
                "invalid_skills_command",
            )

        try:
            self.skill_manager.refresh()
        except Exception as exc:
            return None, None, self._command_result(
                f"Cannot refresh skills: {exc}",
                False,
                str(exc),
            )

        if kind == "list":
            return None, None, self._command_result(
                self.skill_manager.format_skills(),
                True,
            )

        if not task:
            return None, None, self._command_result(
                f"Skill /{name} requires a task. Usage: /{name} <task>",
                False,
                "missing_skill_task",
            )

        skill = self.skill_manager.get_skill(name)
        if skill is None:
            diagnostics = self.skill_manager.diagnostics_for(name)
            if diagnostics:
                details = "\n".join(
                    f"- {item.message}\n  Path: {item.path}"
                    for item in diagnostics
                )
                answer = f"Cannot invoke /{name}.\n\n{details}"
            else:
                answer = (
                    f"Unknown skill: /{name}\n\n"
                    "Use /skills to list available skills."
                )
            return None, None, self._command_result(
                answer,
                False,
                "skill_unavailable",
            )

        return skill, task, None

    def run(self, user_message: str, clear_history: bool = False,
            stream: bool = False, cancel_event=None,
            execution_mode: str = "auto"):
        """
        执行一次 Agent 对话。Agent 公开的唯一入口。

        执行流程（Agent Harness 核心循环）:
          1. 可选：清空历史消息（clear_history=True）
          2. 解析显式 Skill 命令；直接命令无需调用模型
          3. 构建完整系统提示词（包含工具/当前 Skill/记忆）
          4. 将清理后的用户任务追加到对话历史
          5. 进入循环:
             a. 检查步数是否超限 → 超限则截断返回
             b. 调用 LLM.chat(messages, tools, stream=stream)
             c. 解析回复 → text / tool_calls / thinking
             d. 记录本轮操作 (AgentAction)
             e. 如果没有 tool_calls → 这是最终答案 → 退出循环
             f. 如果有 tool_calls → 逐个执行 → 追加工具结果到 messages → 回到 a
          6. 打包 AgentResult 返回（非流式）/ yield done 事件（流式）

        入参:
            user_message:   用户的问题或指令文本
            clear_history:  是否清空历史，开始新对话
            stream:         False → 返回 AgentResult（等全部执行完）
                            True  → 生成器模式，逐块产出 dict（打字机效果）
            cancel_event:   可选的线程安全取消信号；不传时保持原有行为

        返回:
            stream=False: AgentResult 对象
            stream=True:  Generator，逐块 yield 事件 dict:
                          {"type":"text","content":"你"}   → 文字增量
                          {"type":"tool_start","name":"bash"} → 开始执行工具
                          {"type":"tool_end","name":"bash","output":"..."} → 工具执行完成
                          {"type":"done","final_answer":"..."} → 对话结束
        """
        if execution_mode not in {"auto", "lightweight", "full"}:
            raise ValueError("execution_mode 必须是 auto、lightweight 或 full")
        # ── 初始化本轮运行 ──
        self.actions = []                     # 清空本轮操作记录
        total_input_tokens = 0                # 累计输入 token
        total_output_tokens = 0               # 累计输出 token

        # 可选：清除历史
        if clear_history:
            self.messages = []

        # ── 解析本轮显式 Skill 命令 ──
        active_skill = None
        effective_user_message = user_message
        parsed_command = self._parse_skill_command(user_message)
        if parsed_command is not None:
            active_skill, effective_user_message, direct_result = (
                self._resolve_skill_command(parsed_command)
            )
            if direct_result is not None:
                if stream:
                    yield {"type": "done", "result": direct_result}
                    return
                return direct_result

        lightweight_mode = (
            execution_mode == "lightweight"
            or (
                execution_mode == "auto"
                and self._is_lightweight_question(
                    effective_user_message,
                    active_skill=active_skill,
                )
            )
        )
        if active_skill is not None:
            lightweight_mode = False

        # ── 构建系统提示词并放入 messages[0] ──
        full_prompt = self._build_system_prompt(
            effective_user_message,
            active_skill=active_skill,
            include_tools=not lightweight_mode,
            include_memory=not lightweight_mode,
        )
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = full_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": full_prompt})

        # ── 追加用户消息 ──
        self.messages.append(
            {"role": "user", "content": effective_user_message}
        )
        self._trim_and_flush()
        self._threshold_flush()
        self._overflow_flush()

        # ── 准备 tools 定义 ──
        tool_definitions = None
        if self.tools and not lightweight_mode:
            tool_definitions = [t.to_definition() for t in self.tools]

        model_options = {}
        if lightweight_mode and isinstance(getattr(self.model, "config", None), dict):
            model_options["thinking"] = {"type": "disabled"}

        if lightweight_mode:
            logger.info("[Agent] Lightweight mode enabled")

        # ── 主循环 ──
        final_text = ""                       # 最终答案文本
        last_model_text = ""                   # 最近一次模型返回的文字
        step = 0                              # 当前步数
        consecutive_tool_failures: Dict[str, int] = {}
        last_tool_call_signature = None
        consecutive_duplicate_calls = 0

        while step < self.max_steps:
            if _cancel_requested(cancel_event):
                cancelled_result = self._build_cancelled_result(
                    final_text,
                    step,
                    total_input_tokens,
                    total_output_tokens,
                )
                if stream:
                    yield {"type": "stopped", "result": cancelled_result}
                    return
                return cancelled_result
            step += 1
            logger.info("[Agent] Step %d/%d", step, self.max_steps)

            # ── 调用 LLM ──
            try:
                if _cancel_requested(cancel_event):
                    cancelled_result = self._build_cancelled_result(
                        final_text,
                        step - 1,
                        total_input_tokens,
                        total_output_tokens,
                    )
                    if stream:
                        yield {"type": "stopped", "result": cancelled_result}
                        return
                    return cancelled_result
                if stream:
                    # 流式模式: 一次遍历同时完成"产出给用户"+"收集完整响应"
                    response = yield from self._process_stream_response(
                        self.model.chat(messages=self.messages,
                                        tools=tool_definitions, stream=True,
                                        **model_options),
                        step,
                        cancel_event=cancel_event,
                    )
                else:
                    response = self.model.chat(
                        messages=self.messages,
                        tools=tool_definitions,
                        stream=False,
                        **model_options,
                    )
            except Exception as e:
                logger.error("[Agent] LLM call failed at step %d: %s", step, e)
                error_result = AgentResult(
                    final_answer=f"抱歉，调用 AI 模型时出错了: {e}",
                    step_count=step,
                    actions=self.actions,
                    success=False,
                    error=str(e),
                    total_usage={"input": total_input_tokens, "output": total_output_tokens},
                )
                if stream:
                    yield {"type": "done", "result": error_result}
                    return
                return error_result

            # ── 累计 token 用量 ──
            usage = response.get("usage", {})
            total_input_tokens += usage.get("input", 0)
            total_output_tokens += usage.get("output", 0)

            if response.get("cancelled"):
                cancelled_result = self._build_cancelled_result(
                    response.get("text", ""),
                    step,
                    total_input_tokens,
                    total_output_tokens,
                )
                if stream:
                    yield {"type": "stopped", "result": cancelled_result}
                    return
                return cancelled_result

            # ── 判断动作类型 ──
            tool_calls = response.get("tool_calls", [])
            text = response.get("text", "")
            thinking = response.get("thinking", "")
            last_model_text = text

            if tool_calls:
                action_type = AgentActionType.TOOL_USE
            else:
                action_type = AgentActionType.FINAL_ANSWER

            # ── 记录本轮操作 ──
            action = AgentAction(
                action_type=action_type,
                content=text,
                thinking=thinking,
                tool_calls=tool_calls,
                usage={"input": usage.get("input", 0), "output": usage.get("output", 0)},
                step=step,
            )
            self.actions.append(action)
            logger.info("[Agent] Step %d: %s (text=%d chars, tool_calls=%d)",
                        step, action_type.value, len(text), len(tool_calls))

            # ── 分支 1: 最终答案 → 退出循环 ──
            if action_type == AgentActionType.FINAL_ANSWER:
                final_text = text
                self.messages.append({"role": "assistant", "content": text})
                logger.info("[Agent] Task complete at step %d", step)
                break

            # ── 分支 2: 工具调用 → 执行工具 → 追加结果 ──
            assistant_msg = {
                "role": "assistant",
                "content": text or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            # 逐个执行工具
            for tool_index, tc in enumerate(tool_calls):
                if _cancel_requested(cancel_event):
                    self._append_skipped_tool_messages(
                        tool_calls,
                        tool_index,
                        "the Agent run was stopped",
                    )
                    cancelled_result = self._build_cancelled_result(
                        text,
                        step,
                        total_input_tokens,
                        total_output_tokens,
                    )
                    if stream:
                        yield {"type": "stopped", "result": cancelled_result}
                        return
                    return cancelled_result
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_call_id = tc.get("id", "")

                tool_signature = self._tool_call_signature(tool_name, tool_args)
                is_duplicate_call = (
                    tool_signature == last_tool_call_signature
                    and consecutive_duplicate_calls >= _MAX_CONSECUTIVE_DUPLICATE_TOOL_CALLS - 1
                )
                if is_duplicate_call:
                    duplicate_output = (
                        f"Error: tool '{tool_name}' was called repeatedly with "
                        "the same arguments. Execution stopped."
                    )
                    action.tool_result = ToolResult(
                        tool_name=tool_name,
                        input_params=tool_args,
                        output=duplicate_output,
                        success=False,
                        error="duplicate_tool_call",
                        duration_ms=0,
                    )
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": self._prepare_tool_output(duplicate_output),
                    })
                    self._append_skipped_tool_messages(
                        tool_calls,
                        tool_index + 1,
                        "duplicate tool calls were detected",
                    )
                    logger.warning(
                        "[Agent] Tool %s repeated with identical arguments; stopping",
                        tool_name,
                    )
                    duplicate_result = self._build_duplicate_tool_call_result(
                        tool_name,
                        step,
                        total_input_tokens,
                        total_output_tokens,
                    )
                    if stream:
                        yield {"type": "done", "result": duplicate_result}
                        return
                    return duplicate_result

                if stream:
                    yield {"type": "tool_start", "name": tool_name, "args": tool_args}

                tool_instance = self._find_tool(tool_name)
                if tool_instance is None:
                    tool_output = f"Error: Unknown tool '{tool_name}'"
                    tool_success = False
                    tool_error = f"Unknown tool: {tool_name}"
                    elapsed_ms = 0
                else:
                    try:
                        start = time.time()
                        tool_output = tool_instance.execute(tool_args)
                        elapsed_ms = (time.time() - start) * 1000
                        tool_success = not self._tool_output_indicates_failure(
                            tool_output
                        )
                        tool_error = None if tool_success else str(tool_output)
                        if tool_success:
                            logger.info(
                                "[Agent] Tool %s executed in %.0fms",
                                tool_name,
                                elapsed_ms,
                            )
                        else:
                            logger.warning(
                                "[Agent] Tool %s returned a failure result: %s",
                                tool_name,
                                str(tool_output)[:200],
                            )
                    except Exception as e:
                        tool_output = f"Error executing {tool_name}: {e}"
                        tool_success = False
                        tool_error = str(e)
                        elapsed_ms = 0
                        logger.error("[Agent] Tool %s failed: %s", tool_name, e)

                prepared_tool_output = self._prepare_tool_output(tool_output)
                if stream:
                    yield {"type": "tool_end", "name": tool_name,
                           "output": prepared_tool_output, "success": tool_success}

                action.tool_result = ToolResult(
                    tool_name=tool_name,
                    input_params=tool_args,
                    output=tool_output,
                    success=tool_success,
                    error=tool_error,
                    duration_ms=elapsed_ms,
                )

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": prepared_tool_output,
                })

                if tool_success:
                    consecutive_tool_failures.pop(tool_name, None)
                    if tool_signature == last_tool_call_signature:
                        consecutive_duplicate_calls += 1
                    else:
                        last_tool_call_signature = tool_signature
                        consecutive_duplicate_calls = 1
                else:
                    last_tool_call_signature = None
                    consecutive_duplicate_calls = 0
                    failure_count = consecutive_tool_failures.get(tool_name, 0) + 1
                    consecutive_tool_failures[tool_name] = failure_count
                    if failure_count >= _MAX_CONSECUTIVE_TOOL_FAILURES:
                        self._append_skipped_tool_messages(
                            tool_calls,
                            tool_index + 1,
                            "the tool failure limit was reached",
                        )
                        logger.warning(
                            "[Agent] Tool %s failed %d consecutive times; stopping",
                            tool_name,
                            failure_count,
                        )
                        failure_result = AgentResult(
                            final_answer=(
                                f"工具 {tool_name} 已连续失败 {failure_count} 次，"
                                "已停止继续尝试。请检查命令、网络或简化任务后重试。"
                            ),
                            step_count=step,
                            actions=self.actions,
                            success=False,
                            error="tool_failure_limit",
                            total_usage={
                                "input": total_input_tokens,
                                "output": total_output_tokens,
                            },
                        )
                        if stream:
                            yield {"type": "done", "result": failure_result}
                            return
                        return failure_result

                if _cancel_requested(cancel_event):
                    self._append_skipped_tool_messages(
                        tool_calls,
                        tool_index + 1,
                        "the Agent run was stopped",
                    )
                    cancelled_result = self._build_cancelled_result(
                        text,
                        step,
                        total_input_tokens,
                        total_output_tokens,
                    )
                    if stream:
                        yield {"type": "stopped", "result": cancelled_result}
                        return
                    return cancelled_result

        else:
            logger.warning("[Agent] Reached max steps (%d), stopping", self.max_steps)
            result = self._build_max_steps_result(
                last_model_text,
                step,
                total_input_tokens,
                total_output_tokens,
            )
            if stream:
                yield {"type": "done", "result": result}
                return
            return result

        # ── 打包返回 ──
        result = AgentResult(
            final_answer=final_text,
            step_count=step,
            actions=self.actions,
            success=True,
            total_usage={"input": total_input_tokens, "output": total_output_tokens},
        )
        if self.memory_manager:
            auto_refine = getattr(self.memory_manager, '_run_auto_refine', None)
            if callable(auto_refine):
                try:
                    from config import conf
                    auto_refine(conf().get('memory_auto_refine_enabled', False))
                except Exception as exc:
                    logger.warning('[Agent] Auto refine check failed: %s', exc)
        if stream:
            yield {"type": "done", "result": result}
        else:
            return result

    def _process_stream_response(self, stream_gen, step: int, cancel_event=None):
        """
        处理流式 LLM 响应：一次遍历，同时完成两件事。
          ① yield 事件给调用方（前端实时显示）
          ② 拼接完整响应 dict 供 Agent 循环使用

        yield 的事件类型:
          {"type":"text","content":"你好"} → 文字增量，前端追加显示
          {"type":"thinking","content":"..."} → 模型思考过程（可选展示）
          {"type":"tool_call","name":"bash","args":{...}} → 模型请求调工具

        返回值（通过 return，由 yield from 捕获）:
          完整响应 dict，格式和非流式 chat() 一致

        入参:
            stream_gen: LLM.chat(stream=True) 返回的生成器
            step: 当前步数
            cancel_event: 可选的线程安全取消信号

        yield:
            用户可见事件 dict

        返回:
            完整响应 dict（包含 text/tool_calls/thinking/usage/cancelled）
        """
        text_parts = []                     # 累积文字碎片
        thinking_parts = []                 # 累积思考内容
        tool_calls_map = {}                 # {index: {id, name, arguments_str}}
        usage = {"input": 0, "output": 0}   # token 用量

        for chunk in stream_gen:
            if _cancel_requested(cancel_event):
                return {
                    "text": "".join(text_parts),
                    "thinking": "".join(thinking_parts),
                    "tool_calls": [],
                    "usage": usage,
                    "finish_reason": "cancelled",
                    "cancelled": True,
                }

            # ── 错误处理 ──
            if chunk.get("error"):
                raise Exception(chunk.get("message", "Unknown stream error"))

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            # ── 文字增量 → yield 给前端 + 加入累积 ──
            if delta.get("content"):
                content = delta["content"]
                text_parts.append(content)
                yield {"type": "text", "content": content, "step": step}

            # ── 思考增量 → yield（前端可以折叠显示） ──
            if delta.get("reasoning_content"):
                thinking = delta["reasoning_content"]
                thinking_parts.append(thinking)
                yield {"type": "thinking", "content": thinking, "step": step}

            # ── 工具调用增量（流式时分多次到达，按 index 合并） ──
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "name": "", "arguments_str": ""}
                if tc.get("id"):
                    tool_calls_map[idx]["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    tool_calls_map[idx]["name"] = func["name"]
                if func.get("arguments"):
                    tool_calls_map[idx]["arguments_str"] += func["arguments"]

            # ── usage 在最后一个 chunk 中出现 ──
            if chunk.get("usage"):
                u = chunk["usage"]
                usage = {"input": u.get("prompt_tokens", 0),
                         "output": u.get("completion_tokens", 0)}

        # ── 组装 tool_calls ──
        # 流式传输中 arguments 是 JSON 字符串片段，需要拼接后整体解析
        tool_calls = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            try:
                arguments = json.loads(tc["arguments_str"])
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": arguments,
            })

        # ── 返回完整响应（通过 generator return → yield from 捕获） ──
        return {
            "text": "".join(text_parts),
            "thinking": "".join(thinking_parts),
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": "tool_use" if tool_calls else "stop",
            "cancelled": False,
        }

    def clear_history(self):
        """
        清空对话历史，开始新一轮对话。

        调用时机: 用户发送 /clear 或 #清除记忆 命令时。
        """
        self.messages = []
        self.actions = []
        logger.info("[Agent] Conversation history cleared")
