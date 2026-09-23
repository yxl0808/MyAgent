# encoding:utf-8
"""
MyAgent 模型适配层 — Claude 适配器
==================================
接入 Anthropic Claude API（Messages 端点）。

Claude 和 OpenAI 的三个关键差异:
  1. 鉴权:     x-api-key 头  vs  Authorization: Bearer 头
  2. 请求体:   system 是顶层字段，不是 messages[0] 的角色
              tools 使用 input_schema 而非 function.parameters
  3. 响应体:   content 是 list[block]，不是 choices[0].message.content
              每个 block 有 type: "text" / "tool_use" / "thinking"
"""

import json
from typing import Any, Dict, List, Optional

from models.base import BaseLLM


class ClaudeAdapter(BaseLLM):
    """
    Anthropic Claude Messages API 适配器。

    鉴权方式:  x-api-key 头
    API 路径:   /messages
    请求/响应:  Anthropic Messages 格式

    Claude content blocks 类型:
      - text:      {"type": "text", "text": "回复内容"}
      - tool_use:  {"type": "tool_use", "id": "xxx", "name": "bash", "input": {...}}
      - thinking:  {"type": "thinking", "thinking": "思考过程..."}  (extended thinking)
    """

    _api_path = "/messages"

    # ── 钩子方法实现 ──────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """
        Claude 鉴权方式：x-api-key 头（不是 Bearer Token）。
        anthropic-version 声明使用的 API 版本号。

        返回:
            {"x-api-key": "sk-ant-xxx", "anthropic-version": "2023-06-01", "content-type": "application/json"}
        """
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_request_body(self, messages: List[Dict],
                            tools: Optional[List[Dict]],
                            stream: bool, **kwargs) -> Dict[str, Any]:
        """
        组装 Claude Messages API 格式的请求体。

        与 OpenAI 的关键差异:
          1. system 是顶层字段，不在 messages 数组里。
             如果 messages 第一个元素的 role 是 "system"，提取出来放到顶层。
          2. tools 保持 Claude 原生格式:
             {"name":"...", "description":"...", "input_schema":{...}}
          3. max_tokens 是必填字段，Claude 不传会报错。

        入参:
            messages: 标准消息列表（第一个可能是 system role）
            tools:    工具定义列表
            stream:   是否流式
            **kwargs: temperature, max_tokens 等

        返回:
            Claude Messages API 请求体字典
        """
        # ── 提取 system prompt ──
        # Claude 要求 system 是顶层字段，不在 messages 数组中
        claude_messages = []
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                claude_messages.append(msg)

        # ── 组装请求体 ──
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": claude_messages,
            "stream": stream,
        }

        # system 有内容时才传（空字符串 Claude 不接受）
        if system_prompt:
            body["system"] = system_prompt

        # max_tokens: Claude 必填字段，从 config 或 kwargs 取，兜底 4096
        body["max_tokens"] = (
            kwargs.get("max_tokens")
            or self.config.get("max_tokens")
            or 4096
        )

        # temperature: 可选，从 kwargs 或 config 取
        if "temperature" in kwargs or "temperature" in self.config:
            body["temperature"] = kwargs.get("temperature", self.config.get("temperature"))

        # top_p: 可选，从 kwargs 或 config 取
        if "top_p" in kwargs or "top_p" in self.config:
            body["top_p"] = kwargs.get("top_p", self.config.get("top_p"))

        # tools: Claude 原生格式（name/description/input_schema）
        if tools:
            # 如果已经是 Claude 格式（有 input_schema），直接用
            # 否则从 OpenAI 格式转换（取 function 子对象）
            converted = []
            for tool in tools:
                if "input_schema" in tool:
                    converted.append(tool)  # 已经是 Claude 格式
                elif "function" in tool:
                    # OpenAI 格式 → Claude 格式
                    func = tool["function"]
                    converted.append({
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    })
                else:
                    converted.append(tool)
            body["tools"] = converted

        return body

    def _parse_sync_response(self, response_data: Dict) -> Dict[str, Any]:
        """
        将 Claude Messages 响应解析为统一内部格式。

        Claude 原始响应结构:
          {
            "id": "msg_xxx",
            "model": "claude-sonnet-4-6-20250514",
            "stop_reason": "end_turn" | "tool_use",
            "content": [
              {"type": "text", "text": "我可以帮你..."},
              {"type": "tool_use", "id": "toolu_xxx", "name": "bash", "input": {"cmd": "ls"}},
              {"type": "thinking", "thinking": "让我想想..."}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50}
          }

        统一内部格式（和 OpenAI 返回一致）:
          {
            "text": "...",
            "tool_calls": [...],
            "usage": {"input": N, "output": N},
            "finish_reason": "stop" | "tool_use"
          }

        入参:
            response_data: Claude API 返回的完整 JSON dict

        返回:
            统一格式 dict
        """
        # ── 遍历 content blocks 分类提取 ──
        text_parts = []
        tool_calls = []

        for block in response_data.get("content", []):
            btype = block.get("type")

            # 文本块 → 拼接到最终回复
            if btype == "text":
                text_parts.append(block.get("text", ""))

            # 工具调用块 → 标准化为统一格式
            # Claude 的 arguments 已经是 dict（input 字段），不需要 json.loads
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {}),
                })

            # thinking 块 → 不纳入 text（那是模型的内部思考，用户不应看到）
            elif btype == "thinking":
                pass  # 跳过，不加入输出

        # ── 提取 usage ──
        # Claude 字段名: input_tokens / output_tokens → 统一格式: input / output
        usage = response_data.get("usage", {})
        usage_dict = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }

        # ── 转换 stop_reason ──
        # Claude: "end_turn" → 统一: "stop"
        # Claude: "tool_use" → 统一: "tool_use"
        claude_stop = response_data.get("stop_reason", "end_turn")
        finish_reason = "tool_use" if claude_stop == "tool_use" else "stop"

        return {
            "text": "\n".join(text_parts),  # 多个 text block 用换行拼接
            "tool_calls": tool_calls,
            "usage": usage_dict,
            "finish_reason": finish_reason,
        }