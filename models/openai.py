# encoding:utf-8
"""
MyAgent 模型适配层 — OpenAI 适配器
==================================
支持所有 OpenAI 兼容 API（OpenAI 官方、DeepSeek、硅基流动、Groq 等）。

由于 DeepSeek 也使用 OpenAI 兼容格式，DeepSeekAdapter 直接继承本类，
只覆写 _build_headers 中的 api_key 来源和 _api_path。
"""

from typing import Any, Dict, List, Optional

from models.base import BaseLLM


class OpenAIAdapter(BaseLLM):
    """
    OpenAI 兼容 API 适配器。

    鉴权方式:  Bearer Token（Authorization 头）
    API 路径:   /chat/completions
    请求/响应: 标准 OpenAI Chat Completions 格式

    用法:
        adapter = OpenAIAdapter(
            api_key="sk-xxx",
            api_base="https://api.openai.com/v1",
            model="gpt-4o",
            config={"temperature": 0.7},
        )
        result = adapter.chat(messages=[{"role":"user","content":"你好"}])
    """

    # API 端点路径，子类可覆写（如 DeepSeek 也用这个路径）
    _api_path = "/chat/completions"

    # ── 钩子方法实现 ──────────────────────────────────────────

    def _build_headers(self) -> Dict[str, str]:
        """
        OpenAI 鉴权方式：Bearer Token。
        所有 OpenAI 兼容 API（OpenAI、DeepSeek、Groq、硅基流动 等）都使用此格式。

        返回:
            {"Authorization": "Bearer sk-xxx", "Content-Type": "application/json"}
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_body(self, messages: List[Dict],
                            tools: Optional[List[Dict]],
                            stream: bool, **kwargs) -> Dict[str, Any]:
        """
        组装 OpenAI Chat Completions 格式的请求体。

        OpenAI API 标准字段说明:
          model:     必填，模型名
          messages:  必填，对话历史 [{"role":"user","content":"..."}, ...]
          tools:     可选，工具定义列表 [{"type":"function","function":{...}}, ...]
          stream:    可选，是否流式输出
          temperature: 可选，从 kwargs 或 self.config 读取
          max_tokens:  可选，从 kwargs 或 self.config 读取

        返回:
            JSON 请求体字典
        """
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        # ── 可选参数：从 kwargs（调用时传入）或 config（构造时传入）取值 ──
        # 优先级: kwargs > self.config > 不传（让 API 用默认值）
        if "temperature" in kwargs or "temperature" in self.config:
            body["temperature"] = kwargs.get("temperature", self.config.get("temperature"))
        if "max_tokens" in kwargs or "max_tokens" in self.config:
            body["max_tokens"] = kwargs.get("max_tokens", self.config.get("max_tokens"))
        if "top_p" in kwargs or "top_p" in self.config:
            body["top_p"] = kwargs.get("top_p", self.config.get("top_p"))

        # ── 工具定义：只在有工具时才传入 ──
        if tools:
            # 如果已经包含 "type" 字段（OpenAI 格式），直接使用
            # 否则包装为 {"type": "function", "function": {...}} 格式
            converted = []
            for tool in tools:
                if "type" in tool:
                    converted.append(tool)  # 已经是 OpenAI 格式
                else:
                    # Claude 格式 → OpenAI 格式转换
                    converted.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        },
                    })
            body["tools"] = converted

        return body

    def _parse_sync_response(self, response_data: Dict) -> Dict[str, Any]:
        """
        将 OpenAI Chat Completions 响应解析为统一内部格式。

        OpenAI 原始响应结构:
          {
            "choices": [{
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "回复内容",
                "tool_calls": [{
                  "id": "call_xxx",
                  "type": "function",
                  "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}
                }]
              },
              "finish_reason": "stop" | "tool_calls" | "length"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
          }

        统一内部格式（所有模型适配器返回一样）:
          {
            "text": "回复内容",               # str，没有为 ""
            "tool_calls": [...],             # list，没有为 []
            "usage": {"input": N, "output": N},  # dict
            "finish_reason": "stop"          # str
          }

        入参:
            response_data: requests.json() 解析后的完整响应字典

        返回:
            统一格式 dict
        """
        # 取第一个（也是唯一一个）choice
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # 提取文字内容（可能为 None——纯工具调用场景）
        text = message.get("content") or ""

        # 提取工具调用列表并标准化格式
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            # 标准化: arguments 在 OpenAI 里是 JSON 字符串，需要解析为 dict
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                import json
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}  # 解析失败就用空 dict
            else:
                arguments = raw_args  # 已经是 dict（某些兼容 API 直接返回 dict）
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": arguments,
            })

        # 提取 token 用量
        usage = response_data.get("usage", {})
        usage_dict = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        }

        # finish_reason: "stop" 表示正常结束，"tool_calls" 表示模型想调用工具
        finish_reason = choice.get("finish_reason", "stop")

        return {
            "text": text,
            "tool_calls": tool_calls,
            "usage": usage_dict,
            "finish_reason": finish_reason,
        }