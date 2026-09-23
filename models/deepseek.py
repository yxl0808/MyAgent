# encoding:utf-8
"""
MyAgent 模型适配层 — DeepSeek 适配器
====================================
DeepSeek API 完全兼容 OpenAI Chat Completions 格式，
因此本适配器继承 OpenAIAdapter，只追加 Thinking Mode 相关逻辑。

Thinking Mode 简介（DeepSeek V4 系列专有）:
  - 模型在给出回复前先进行内部推理（类似"草稿纸"）
  - 推理过程通过 reasoning_content 字段返回（不展示给最终用户）
  - reasoning_effort 控制推理深度："high"（默认）/ "max"
  - 开启 thinking 后，temperature/top_p/frequency_penalty 等参数被 API 自动忽略
  - **关键约束**: 一旦历史中有 tool_call，后续每条 assistant 消息都必须携带 reasoning_content
"""

from typing import Any, Dict, List, Optional

from models.openai import OpenAIAdapter


class DeepSeekAdapter(OpenAIAdapter):
    """
    DeepSeek API 适配器。
    继承 OpenAIAdapter 的全部方法，只覆写以下 3 个以支持 Thinking Mode:
      - _build_request_body: 追加 thinking / reasoning_effort 字段
      - _parse_sync_response: 提取 reasoning_content
      - _parse_stream: 保留 thinking delta（非文本/工具调用的 delta 类型）
    """

    _api_path = "/chat/completions"

    # ── Thinking Mode 辅助方法 ───────────────────────────────

    def _model_supports_thinking(self) -> bool:
        """
        判断当前模型是否支持显式 thinking 开关。
        DeepSeek V4 系列（deepseek-v4-*）支持，V3/reasoner 不支持（原因不同）。
        """
        return self.model.lower().startswith("deepseek-v4")

    def _is_reasoner_model(self) -> bool:
        """
        deepseek-reasoner (R1) 是"自带思考"的模型——永远内部推理，不需要开关。
        """
        return self.model and "reasoner" in self.model.lower()

    # ── 钩子方法覆写 ─────────────────────────────────────────

    def _build_request_body(self, messages: List[Dict],
                            tools: Optional[List[Dict]],
                            stream: bool, **kwargs) -> Dict[str, Any]:
        """
        在 OpenAI 标准请求体的基础上，追加 DeepSeek Thinking Mode 相关字段。

        逻辑:
          1. 调用父类 (OpenAIAdapter) 组装标准请求体
          2. 如果是 V4 模型则按配置决定是否开启 thinking
          3. 如果是 reasoner (R1) → 不做处理（自带思考）
          4. 开启 thinking 后，移除 temperature/top_p 等被忽略的参数

        thinking 字段格式:
          {"thinking": {"type": "enabled"}}     → 开启思考
          {"thinking": {"type": "disabled"}}    → 不思考
        reasoning_effort 可选值:
          "high" (默认), "max"
        """
        # 先用父类逻辑构建标准请求体
        body = super()._build_request_body(messages, tools, stream, **kwargs)

        # 判断是否开启 thinking
        # 优先级: kwargs 显式传入 > self.config 配置 > 默认开启
        thinking_param = kwargs.pop("thinking", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)

        thinking_active = False

        if self._model_supports_thinking():
            # V4: 默认开启，轻量模式可通过本轮参数临时关闭。
            if thinking_param is None:
                thinking_param = self.config.get(
                    "thinking", {"type": "enabled"},
                )
            body["thinking"] = thinking_param
            thinking_active = thinking_param.get("type") == "enabled"

            if thinking_active:
                # reasoning_effort: kwargs > config > 默认 "high"
                effort = reasoning_effort or self.config.get("reasoning_effort") or "high"
                body["reasoning_effort"] = effort

        elif self._is_reasoner_model():
            # R1 自带思考能力，不需要 thinking 开关
            thinking_active = True
            # R1 对 temperature/top_p 等参数也是静默忽略，清理干净
            for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                body.pop(k, None)

        # ── 开启 thinking 后清理被忽略的参数 ──
        # DeepSeek API 规定：thinking 模式下 temperature/top_p/presence_penalty/frequency_penalty
        # 会被静默忽略，我们主动清理以保持请求干净
        if thinking_active:
            for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                body.pop(k, None)

        return body

    def _parse_sync_response(self, response_data: Dict) -> Dict[str, Any]:
        """
        在 OpenAI 标准响应解析的基础上，追加提取 reasoning_content。

        DeepSeek thinking mode 响应中，message 额外包含:
          "reasoning_content": "这是模型的内部推理过程..."  (可选字段)

        reasoning_content 会被放入统一格式的 "thinking" 字段中，
        供上层 Agent 在消息历史中回传（DeepSeek 要求 tool_call 后必须携带）。

        入参:
            response_data: DeepSeek API 返回的完整 JSON dict

        返回:
            统一格式 dict（比 OpenAI 多一个 "thinking" 字段）
        """
        # 先用父类逻辑解析标准字段 (text, tool_calls, usage, finish_reason)
        result = super()._parse_sync_response(response_data)

        # ── 额外提取 reasoning_content（DeepSeek 专有） ──
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})
        reasoning = message.get("reasoning_content", "")

        # 如果这次响应中包含 tool_calls 且没有 reasoning_content，
        # DeepSeek API 在后续请求中会要求每条 assistant 消息都有此字段。
        # 这里给一个空字符串作为兜底，确保上层 Agent 不会因为缺字段而 400。
        if result["tool_calls"] and not reasoning:
            reasoning = ""

        result["thinking"] = reasoning
        return result
