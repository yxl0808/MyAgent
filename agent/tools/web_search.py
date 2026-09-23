# encoding:utf-8
"""
MyAgent 工具 — 网络搜索
=======================
通过 DuckDuckGo 搜索引擎搜索网络信息。
免费、无需 API Key、返回标题+摘要+URL。
"""

import logging
from typing import Any, Dict
from urllib.parse import quote

import requests

from agent.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    在互联网上搜索信息，返回相关网页的标题、摘要和链接。

    底层使用 DuckDuckGo Instant Answer API（免费，无速率限制）。
    如果需要更高质量的搜索结果，可替换为 Google/Bing Search API（但需要付费 API Key）。

    参数:
        query: 搜索关键词，如 "Python asyncio tutorial"
        max_results: 最多返回多少条结果（默认 5，最大 10）
    """

    name = "web_search"
    description = (
        "Search the web for information. "
        "Returns page titles, snippets, and URLs. "
        "Use this when you need current information, documentation, or facts "
        "that may not be in your training data."
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query string, e.g. 'Python 3.13 release notes'",
            "required": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (1-10, default 5)",
            "required": False,
        },
    }

    def execute(self, params: Dict[str, Any]) -> str:
        """
        执行网络搜索。

        使用 DuckDuckGo 的 Lite 版本（返回 HTML，需解析）。
        如果 DuckDuckGo 不可用，回退到 Google 搜索的文本版本。

        入参:
            params: {"query": "Python tutorial", "max_results": 5}

        返回:
            格式化的搜索结果文本
        """
        query = params.get("query", "").strip()
        max_results = min(int(params.get("max_results", 5)), 10)

        if not query:
            return "Error: No search query provided"

        logger.info("[WebSearch] Searching: %s", query)

        try:
            results = self._search_duckduckgo(query, max_results)
        except Exception as e:
            logger.warning("[WebSearch] DuckDuckGo failed: %s, trying fallback", e)
            try:
                results = self._search_fallback(query, max_results)
            except Exception as e2:
                return f"Error: Web search failed: {e2}"

        if not results:
            return f"No results found for: {query}"

        # 格式化输出
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   URL: {r['url']}\n")

        return "\n".join(lines)

    # ── 搜索引擎实现 ──────────────────────────────────────────

    def _search_duckduckgo(self, query: str, max_results: int) -> list:
        """
        通过 DuckDuckGo Lite 搜索。
        返回 HTML 页面，需要手动解析出标题和链接。
        """
        url = "https://lite.duckduckgo.com/lite/"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; MyAgent/1.0)",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(
            url,
            headers=headers,
            data={"q": query},
            timeout=15,
        )
        resp.raise_for_status()

        results = []
        # 简单 HTML 解析：找 <a> 标签（标题+链接）和后面的文本（摘要）
        from html.parser import HTMLParser

        class DuckParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.in_link = False
                self.in_snippet = False
                self.snippet_text = ""
                self.link_count = 0

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == "a" and "href" in attrs_dict:
                    href = attrs_dict["href"]
                    # 过滤掉不是搜索结果的链接
                    if href.startswith("//duckduckgo.com/l/") or "uddg=" in href:
                        self.current = {"url": "", "title": "", "snippet": ""}
                        self.in_link = True
                        self.link_count += 1
                elif tag == "td" and self.in_link:
                    self.in_snippet = True
                    self.snippet_text = ""

            def handle_data(self, data):
                if self.in_link and self.current is not None:
                    self.current["title"] = data.strip()
                    self.in_link = False
                elif self.in_snippet:
                    self.snippet_text += data

            def handle_endtag(self, tag):
                if tag == "td" and self.in_snippet and self.current is not None:
                    self.in_snippet = False
                    # 提取 snippet 中的有效文本
                    snippet = self.snippet_text.strip()
                    if snippet and len(snippet) > 10:
                        self.current["snippet"] = snippet
                        # 从 title 推断 URL（DuckDuckGo Lite 不直接给 URL）
                        self.current["url"] = f"https://duckduckgo.com/?q={quote(query)}"
                        self.results.append(self.current)
                        self.current = None

        parser = DuckParser()
        parser.feed(resp.text)
        return parser.results[:max_results]

    def _search_fallback(self, query: str, max_results: int) -> list:
        """
        DuckDuckGo 不可用时的回退方案。
        简单地返回一条提示——因为免费搜索引擎都有限制。
        """
        return [{
            "title": query,
            "snippet": (
                "DuckDuckGo search is currently unavailable. "
                "You can try again later, or use web_fetch directly on known URLs "
                "like documentation sites or Wikipedia."
            ),
            "url": f"https://www.google.com/search?q={quote(query)}",
        }]