# encoding:utf-8
"""
MyAgent 工具 — 网页抓取
=======================
抓取指定 URL 的网页内容，提取纯文本。
"""

import html as html_module
import logging
import re
from typing import Any, Dict
from urllib.parse import urlparse

import requests

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebFetchTool(BaseTool):
    """
    抓取网页内容并提取纯文本。

    处理流程:
      1. 发送 HTTP GET 请求
      2. 检测 Content-Type，只处理 text/html
      3. 去除 HTML 标签、脚本、样式
      4. 返回纯文本内容（截断到安全长度）

    安全限制:
      - URL 必须是 http/https 协议
      - 不跟随重定向到不同域名
      - 超时 15 秒
      - 内容截断到 50000 字符（防止上下文爆炸）

    参数:
        url: 要抓取的网址
    """

    name = "web_fetch"
    description = (
        "Fetch a known http/https URL in one request and return its title plus "
        "cleaned main article text. Use this first for ordinary web pages instead "
        "of bash, curl, wget, or PowerShell."
    )
    parameters = {
        "url": {
            "type": "string",
            "description": "The URL to fetch, e.g. 'https://docs.python.org/3/library/asyncio.html'",
            "required": True,
        },
    }

    # 最大返回内容长度（字符数），防止上下文爆炸
    MAX_CONTENT_LENGTH = 16000

    def execute(self, params: Dict[str, Any]) -> str:
        """
        抓取网页并返回纯文本。

        入参:
            params: {"url": "https://example.com"}

        返回:
            清理后的网页文本，或错误信息
        """
        url = params.get("url", "").strip()
        if not url:
            return "Error: No URL provided"

        # ── 安全校验：只允许 http/https ──
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Error: Unsupported URL scheme '{parsed.scheme}'. Only http and https are allowed."

        logger.info("[WebFetch] Fetching: %s", url)

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; MyAgent/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            is_html = (
                "text/html" in content_type
                or "application/xhtml+xml" in content_type
            )
            is_plain_text = "text/plain" in content_type

            # ── 只处理文本类型 ──
            if not is_html and not is_plain_text:
                return (
                    f"Error: Cannot process content type '{content_type}'. "
                    f"This tool only handles HTML and plain text pages.\n"
                    f"Content-Length: {len(resp.content)} bytes"
                )

            raw = self._decode_content(resp)

            # ── HTML → 纯文本 ──
            if is_html:
                title = self._extract_title(raw)
                text = self._html_to_text(raw)
                if not text:
                    return f"Error: Web page did not contain readable text: {url}"
                text = f"URL: {url}\nTitle: {title or '(untitled)'}\n\nContent:\n{text}"
            else:
                text = raw

            # ── 截断过长内容 ──
            if len(text) > self.MAX_CONTENT_LENGTH:
                original_length = len(text)
                text = text[:self.MAX_CONTENT_LENGTH]
                text += f"\n\n... (truncated, original length: {original_length} chars)"

            logger.info("[WebFetch] Fetched %d chars from %s", len(text), url)
            return text

        except requests.Timeout:
            return f"Error: Request timed out after 15 seconds: {url}"
        except requests.HTTPError as e:
            return f"Error: HTTP {e.response.status_code} when fetching {url}"
        except requests.ConnectionError:
            return f"Error: Could not connect to {url}. Check if the URL is correct and the site is accessible."
        except Exception as e:
            logger.error("[WebFetch] Failed: %s", e)
            return f"Error fetching URL: {e}"

    @staticmethod
    def _decode_content(response) -> str:
        """按 HTTP、HTML 声明和编码探测顺序解码响应字节。"""
        content = response.content
        declared_encoding = WebFetchTool._find_declared_encoding(
            content,
            response.headers.get("Content-Type", ""),
        )
        candidates = [
            declared_encoding,
            getattr(response, "apparent_encoding", None),
            "utf-8",
            "gb18030",
        ]
        for encoding in dict.fromkeys(item for item in candidates if item):
            try:
                return content.decode(encoding)
            except (LookupError, UnicodeDecodeError):
                continue
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _find_declared_encoding(content: bytes, content_type: str) -> str | None:
        """从 HTTP Content-Type 或 HTML 头部提取显式字符集声明。"""
        match = re.search(
            r"charset\s*=\s*[\"']?([\w.-]+)",
            content_type,
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        header = content[:8192].decode("ascii", errors="ignore")
        match = re.search(
            r"(?:charset|encoding)\s*=\s*[\"']?([\w.-]+)",
            header,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _extract_title(html: str) -> str:
        """提取页面标题并转换为可读纯文本。"""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
        return WebFetchTool._normalize_text(match.group(1)) if match else ""

    @staticmethod
    def _html_to_text(html: str) -> str:
        """提取 main/article 正文并移除页面框架和 HTML 标记。"""
        primary_match = re.search(
            r"<(?:main|article)\b[^>]*>(.*?)</(?:main|article)>",
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if primary_match:
            html = primary_match.group(1)
        for tag in (
            "aside", "footer", "form", "header", "iframe", "nav", "noscript",
            "script", "select", "style", "svg", "template",
        ):
            html = re.sub(
                rf"<{tag}\b[^>]*>.*?</{tag}>",
                "",
                html,
                flags=re.DOTALL | re.IGNORECASE,
            )
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        for tag in (
            "article", "blockquote", "br", "dd", "div", "dl", "dt", "h1", "h2",
            "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p", "section",
            "table", "td", "th", "tr", "ul",
        ):
            html = re.sub(rf"</?{tag}\b[^>]*>", "\n", html, flags=re.IGNORECASE)
        return WebFetchTool._normalize_text(re.sub(r"<[^>]+>", "", html))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """解码 HTML 实体、统一空白并保留段落边界。"""
        text = html_module.unescape(text).replace("\xa0", " ")
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
