# encoding:utf-8
"""
MyAgent 模型适配层 — 基类模块
===============================
职责：
  1. 定义所有 LLM 适配器的统一接口（BaseLLM 抽象基类）
  2. 提供通用的 HTTP 请求、重试、流式解析逻辑（子类直接继承）
  3. 通过 Template Method 模式，子类只需覆写 3-4 个钩子方法即可接入新模型

设计模式：Template Method（模板方法）
  - 骨架方法 chat() 在基类中定义，控制整个调用流程
  - 钩子方法 _build_* / _parse_* 由子类实现，填写模型特定的细节
  - 子类不修改流程，只填充差异

使用方式（子类覆写示例）:
  class MyModel(BaseLLM):
      def _build_headers(self):          # 必须覆写：鉴权头
          return {"Authorization": f"Bearer {self.api_key}"}

      def _build_url(self, path):        # 必须覆写：API 端点 URL
          return f"{self.api_base}{path}"

      def _build_request_body(self, messages, tools, stream, **kwargs):
          return {                        # 必须覆写：组装请求体
              "model": self.model,
              "messages": messages,
              "tools": tools,
              "stream": stream,
          }

      def _parse_sync_response(self, response_data):
          return {                        # 必须覆写：解析同步响应
              "text": response_data["choices"][0]["message"]["content"],
              "tool_calls": ...,
          }
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional

import requests

# 本模块的 logger
logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    所有 LLM 适配器的抽象基类。

    属性（实例变量）:
        api_key  (str): 从 config 读取的 API 密钥
        api_base (str): API 端点基础 URL（如 https://api.openai.com/v1）
        model    (str): 模型名（如 gpt-4o / claude-sonnet-4-6）
        config   (dict): 额外的模型特定配置（temperature, max_tokens 等）
        proxy    (str|None): HTTP 代理地址

    公开方法:
        chat(messages, tools, stream) → dict | Generator
            所有 Agent 代码调用模型的唯一入口。
            - 入参 messages 是标准化的消息列表: [{"role": "user", "content": "..."}, ...]
            - 入参 tools 是工具定义列表: [{"name": "bash", "description": "...", ...}]
            - stream=True 返回生成器，逐块产出 delta dict
            - stream=False 返回单次 dict: {"text": "...", "tool_calls": [...], "usage": {...}}
    """

    def __init__(self, api_key: str, api_base: str, model: str,
                 config: Optional[Dict[str, Any]] = None,
                 proxy: Optional[str] = None):
        """
        初始化模型适配器实例。

        调用时机：由工厂函数 create_model() 在启动时调用一次，
        创建后作为全局单例被所有 Agent 请求复用。

        入参:
            api_key:  API 密钥字符串（从 config.json 读取）
            api_base: API 基础 URL（从 config.json 读取）
            model:    模型名称（如 "gpt-4o" / "claude-sonnet-4-6" / "deepseek-chat"）
            config:   模型特定参数的可选字典，例如:
                      {"temperature": 0.7, "max_tokens": 8192, "top_p": 0.9}
            proxy:    HTTP 代理 URL（如 "http://127.0.0.1:7890"），不需要时传 None
        """
        # 核心配置：所有子类都用这三个来构造请求
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")  # 去掉尾部斜杠，后面拼接时统一加
        self.model = model

        # 额外配置：temperature, max_tokens 等，子类在 _build_request_body 中使用
        self.config = config or {}

        # HTTP 代理：requests 库需要的格式是 {"http": url, "https": url}
        self._proxies = {"http": proxy, "https": proxy} if proxy else None

    # ── 工具方法（子类可用，通常不需要覆写） ─────────────────

    def _http_post(self, url: str, headers: Dict[str, str],
                   body: Dict[str, Any], stream: bool = False,
                   timeout: int = 120) -> requests.Response:
        """
        发送 HTTP POST 请求到模型 API，失败时自动重试（最多 2 次）。

        重试策略（每类错误的处理方式不同）:
          429 (Rate Limit):   等 20 秒后重试 — 请求太快，API 限流了
          5xx (Server Error): 等 3 秒后重试 — 服务端暂时故障，等一下可能恢复
          401 (Unauthorized): 不重试 — API Key 错了，重试也没用
          超时 (Timeout):     等 5 秒后重试 — 网络波动
          其他错误:           不重试 — 未知问题，不要浪费时间

        入参:
            url:     完整 API 端点 URL，如 "https://api.openai.com/v1/chat/completions"
            headers: 请求头字典（包含鉴权和 Content-Type）
            body:    请求体字典（model, messages, tools, stream 等）
            stream:  是否启用流式响应（True 时 response 不立即下载完整 body）
            timeout: 请求超时秒数，默认 120

        返回:
            requests.Response 对象（成功时 status_code < 400）

        异常:
            requests.HTTPError: 重试 2 次后仍失败，或遇到不可重试的错误（401 等）
        """
        # 最多重试 2 次（总共发 3 次请求）
        max_retries = 2
        last_error = None  # 记录最后一次异常，用于最终抛出
        proxies = self._proxies  # 实例级别代理配置

        for attempt in range(max_retries + 1):  # 0, 1, 2（第一次 + 两次重试）
            try:
                # 发送 POST 请求
                # json=body 自动序列化为 JSON + 设置 Content-Type
                # stream=True 时不立即下载响应体（流式场景）
                response = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=timeout,
                    proxies=proxies,
                    stream=stream,
                )

                # 状态码 < 400 表示成功，直接返回
                if response.status_code < 400:
                    return response

                # ── 非 2xx/3xx，按状态码分类处理 ──
                last_error = requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )

                # 429: 请求频率超限，等一会再试
                if response.status_code == 429:
                    if attempt < max_retries:
                        logger.warning("[LLM] Rate limited (429), retrying in 20s (attempt %d/%d)",
                                       attempt + 1, max_retries)
                        time.sleep(20)
                        continue

                # 5xx: 服务端暂时故障，短等后重试
                elif response.status_code >= 500:
                    if attempt < max_retries:
                        logger.warning("[LLM] Server error (%d), retrying in 3s (attempt %d/%d)",
                                       response.status_code, attempt + 1, max_retries)
                        time.sleep(3)
                        continue

                # 401 / 403 / 4xx（非 429）: 鉴权或请求格式错误，重试无意义
                else:
                    logger.error("[LLM] Client error (%d), not retrying: %s",
                                 response.status_code, response.text[:200])
                    break  # 跳出 for 循环，不重试

            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning("[LLM] Network error (%s), retrying in 5s (attempt %d/%d)",
                                   type(e).__name__, attempt + 1, max_retries)
                    time.sleep(5)
                    continue

        # 重试次数用完了还没成功，抛出最后的异常
        raise last_error

    def _build_url(self, path: str) -> str:
        """拼接模型 API 的完整请求地址。"""
        return f"{self.api_base}{path}"

    # ── 公开方法（Agent 调用的唯一入口） ────────────────────────

    def chat(self, messages: List[Dict[str, Any]],
             tools: Optional[List[Dict[str, Any]]] = None,
             stream: bool = False, **kwargs):
        """
        Agent 调用 LLM 的唯一入口。支持流式和非流式两种模式。

        这是 Template Method 模式的骨架方法——控制调用流程不变，
        具体细节（请求头、请求体、响应解析）由子类的钩子方法提供。

        调用流程（非流式）:
          1. 子类覆写的 _build_headers() → 鉴权头
          2. 子类覆写的 _build_request_body() → 请求体
          3. 基类的 _build_url() → 拼接完整 URL
          4. 基类的 _http_post() → 发请求 + 重试
          5. 子类覆写的 _parse_sync_response() → 解析为统一格式 dict

        调用流程（流式）:
          1-4 同上，但 _http_post 拿到 Response 对象后不立即读 body
          5. 基类的 _parse_stream() → 逐 chunk 解析 SSE 流
          6 每个 chunk 原样 yield 给上层（上层自己拼装最终结果）

        入参:
            messages: 标准化消息列表 [{"role":"user","content":"你好"}, ...]
                      支持纯文本和图片（image_url）两种 content 格式
            tools:    工具定义列表 [{"name":"bash","description":"...","input_schema":{...}}, ...]
                      为 None 或 [] 表示纯对话，不启用工具
            stream:   True → 返回生成器，逐块产出 SSE chunk dict
                      False → 返回单次 dict: {"text":"...","tool_calls":[...],"usage":{...}}
            **kwargs: 透传给 _build_request_body（temperature, max_tokens 等）

        返回:
            非流式: dict — {"text": str, "tool_calls": list, "usage": dict, "finish_reason": str}
            流式:   Generator — 逐个 yield SSE chunk dict:
                    {"choices":[{"delta":{"content":"你好"},"index":0}]}

        异常:
            requests.HTTPError: 请求失败（_http_post 重试耗尽后抛出）
            json.JSONDecodeError: 响应体 JSON 解析失败
        """
        # ── 步骤 1-2: 构造请求 ──
        # 请求头和请求体由子类分别实现，适配不同 API 格式
        headers = self._build_headers()
        body = self._build_request_body(messages, tools, stream, **kwargs)
        # 子类指定自己的 API 路径（如 "/chat/completions" 或 "/messages"）
        url = self._build_url(self._api_path)

        # ── 步骤 3: 发送请求 ──
        # _http_post 内部有重试逻辑（429/5xx/超时自动重试）
        response = self._http_post(
            url,
            headers,
            body,
            stream=stream,
            timeout=self.config.get("request_timeout", 120),
        )

        # ── 步骤 4: 解析响应 ──
        if stream:
            # 流式: 直接返回 SSE 解析生成器，上层逐块消费
            return self._parse_stream(response)
        else:
            # 非流式: 解析完整响应体 JSON，转为统一格式 dict
            data = response.json()
            return self._parse_sync_response(data)

    def _parse_stream(self, response) -> Generator:
        """
        解析 SSE（Server-Sent Events）流式响应，逐块产出 dict。

        SSE 协议简介（子集）:
          - 每个事件以一行 "data: <json>" 开头
          - 事件之间用空行（"\\n\\n"）分隔
          - 流结束标志是 "data: [DONE]"
          - 示例:
              data: {"choices":[{"delta":{"content":"你好"}}]}
              <空行>
              data: {"choices":[{"delta":{"content":"世界"}}]}
              <空行>
              data: [DONE]

        为什么不用 requests 的 iter_lines():
          iter_lines(decode_unicode=True) 会在网络包边界处把多字节 UTF-8 字符切碎，
          导致 JSON 解析失败。正确做法是累积原始字节，找到完整事件边界后再解码。

        入参:
            response: requests.post(stream=True) 返回的 Response 对象

        产出（生成器）:
            每个 yield 返回一个 dict，即 API 返回的单个 SSE chunk
            示例: {"choices": [{"delta": {"content": "你好"}, "index": 0}]}

        异常情况:
            如果遇到 JSON 解析失败的 chunk，记录警告后跳过（不中断流）
        """
        #网络字节流 → 累积到缓冲区 → 找到完整事件边界（\n）→ 解码 → 提取 data: 行 → JSON 解析 → yield dict

        # 字节缓冲区：累积网络包中的原始字节，直到凑够一个完整事件
        buf = b""
        # 遍历网络响应体中的每一段原始字节（chunk_size=None 表示按接收到的包大小来）
        for raw_bytes in response.iter_content(chunk_size=None, decode_unicode=False):
            if not raw_bytes:  # 空包，跳过
                continue
            buf += raw_bytes

            # 在缓冲区中循环查找完整事件（以 "\\n\\n" 或 "\\r\\n\\r\\n" 结尾）
            while True:
                # 查找最早出现的事件分隔符位置
                # 三种可能的换行格式: \n\n, \r\r, \r\n\r\n
                idx_double_lf = buf.find(b"\n\n")
                idx_double_cr = buf.find(b"\r\r")
                idx_crlf = buf.find(b"\r\n\r\n")

                # 收集所有找到的分隔符位置（-1 表示没找到）
                positions = [
                    p for p in (idx_double_lf, idx_double_cr, idx_crlf) if p != -1
                ]
                if not positions:  # 还没凑够一个完整事件，继续收字节
                    break

                # 取最早出现的分隔符，确定事件字节段 + 分隔符长度
                end_pos = min(positions)
                if end_pos == idx_crlf:
                    term_len = 4  # \r\n\r\n 占 4 字节
                else:
                    term_len = 2  # \n\n 或 \r\r 占 2 字节

                event_bytes = buf[:end_pos]           # 事件内容（不含分隔符）
                buf = buf[end_pos + term_len:]        # 剩余字节放回缓冲区

                # 将事件字节解码为 UTF-8 字符串
                # errors="replace" 是最后的安全阀：遇到畸形的字节用 � 替代，不崩溃
                try:
                    event_text = event_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    event_text = event_bytes.decode("utf-8", errors="replace")#占位符

                # 逐行解析事件文本，提取 "data:" 开头的行
                data_lines = []
                for line in event_text.splitlines():
                    # 跳过空行和注释行（以 ":" 开头的是 SSE 注释）
                    if not line or line.startswith(":"):
                        continue
                    # 以 "data:" 开头 → 提取后面的 JSON
                    if line.startswith("data:"):
                        # "data: <json>" → 去掉前缀 "data:"（5 个字符）
                        value = line[5:]
                        # SSE 规范：冒号后如果有空格，空格属于格式不是内容
                        if value.startswith(" "):
                            value = value[1:]
                        data_lines.append(value)

                if not data_lines:  # 这个事件里没有 data 行（如 "event:" 或 "id:"），跳过
                    continue

                # 标准规定：多条 data 行用 "\\n" 拼接
                data_str = "\n".join(data_lines)

                # 流结束标志
                if data_str.strip() == "[DONE]":
                    return

                # 尝试解析为 JSON dict
                try:
                    chunk = json.loads(data_str)
                    yield chunk
                except json.JSONDecodeError:
                    logger.debug("[LLM] Skip malformed SSE chunk: %s", data_str[:100])
                    continue

    # ── 抽象方法（子类必须覆写） ────────────────────────────
    # 每个模型提供商有不同的鉴权方式、请求体格式、响应体格式。
    # 这些差异通过下面 3 个抽象方法的形式化，强制子类实现。

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """
        构造 HTTP 请求头（鉴权头 + Content-Type）。

        必须覆写，因为每个模型的鉴权方式不同：
          OpenAI/DeepSeek: {"Authorization": "Bearer sk-xxx", "Content-Type": "application/json"}
          Claude:          {"x-api-key": "sk-ant-xxx", "anthropic-version": "2023-06-01", ...}

        返回:
            HTTP 请求头字典
        """
        ...

    @abstractmethod
    def _build_request_body(self, messages: List[Dict], tools: Optional[List[Dict]],
                            stream: bool, **kwargs) -> Dict[str, Any]:
        """
        将标准化的 messages 和 tools 组装为模型特定的请求体 JSON。

        必须覆写，因为每个模型的请求体格式不同：
          OpenAI:  {"model": "...", "messages": [...], "tools": [...], "stream": true}
          Claude:  {"model": "...", "system": "...", "messages": [...], "tools": [...], "max_tokens": 4096}

        入参:
            messages: 标准消息列表 [{"role":"user","content":"..."}, ...]
                      可能是纯文本 content，也可能包含 image_url 等多模态 content
            tools:    工具定义列表，格式为 [{"name":"bash","description":"...","input_schema":{...}}, ...]
                      为 None 或空列表时表示不启用工具
            stream:   是否启用流式输出
            **kwargs: 额外参数（temperature, max_tokens 等），由调用方传入

        返回:
            requests.post(json=...) 可直接使用的请求体字典
        """
        ...

    @abstractmethod
    def _parse_sync_response(self, response_data: Dict) -> Dict[str, Any]:
        """
        将模型的原始 HTTP 响应解析为统一的内部格式。

        必须覆写，因为每个模型的响应体格式不同：
          OpenAI:  {"choices":[{"message":{"content":"...","tool_calls":[...]}}], "usage":{...}}
          Claude:  {"content":[{"type":"text","text":"..."},{"type":"tool_use",...}], "usage":{...}}

        入参:
            response_data: 模型 API 返回的完整 JSON dict（已经过 requests.json() 解析）

        返回:
            统一格式的字典，必须包含以下字段:
              "text"       (str):  模型的文字回复内容，没有则为空字符串
              "tool_calls" (list): 工具调用列表，没有则为空列表 []
                                  每个元素: {"id":"...", "name":"...", "arguments":{...}}
              "usage"      (dict): token 用量 {"input": N, "output": N}
              "finish_reason" (str): 结束原因，如 "stop" / "tool_use"
        """
        ...