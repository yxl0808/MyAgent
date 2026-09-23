import unittest
from unittest.mock import patch

from agent.agent import Agent
from agent.tools.bash import BashTool
from agent.tools.web_fetch import WebFetchTool


class FakeResponse:
    """提供字节内容和编码提示的最小 HTTP 响应替身。"""

    def __init__(self, content, content_type, apparent_encoding):
        """保存网页字节、内容类型和推断编码。"""
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.apparent_encoding = apparent_encoding
        self.encoding = None

    def raise_for_status(self):
        """模拟成功 HTTP 响应。"""


class WebFetchToolTest(unittest.TestCase):
    def test_execute_decodes_declared_chinese_page_and_extracts_article_body(self):
        """验证网页工具自动解码中文并排除导航、脚本和页脚噪声。"""
        html = """
        <html>
          <head>
            <meta charset="gb18030">
            <title>研究生招生通知</title>
            <style>.hidden { display: none; }</style>
          </head>
          <body>
            <header>学校门户</header>
            <nav>首页 | 通知 | 联系我们</nav>
            <main>
              <article>
                <h1>2026 年招生公告</h1>
                <p>第一段内容&nbsp;包含中文与 &amp; HTML 实体。</p>
                <p>第二段内容。</p>
              </article>
            </main>
            <footer>版权所有</footer>
            <script>window.hidden = true;</script>
          </body>
        </html>
        """.encode("gb18030")
        response = FakeResponse(html, "text/html", "gb18030")

        with patch("agent.tools.web_fetch.requests.get", return_value=response):
            result = WebFetchTool().execute({"url": "https://example.com/admission"})

        self.assertIn("Title: 研究生招生通知", result)
        self.assertIn("2026 年招生公告", result)
        self.assertIn("第一段内容 包含中文与 & HTML 实体。", result)
        self.assertIn("第二段内容。", result)
        self.assertNotIn("学校门户", result)
        self.assertNotIn("首页 | 通知", result)
        self.assertNotIn("版权所有", result)
        self.assertNotIn("window.hidden", result)

    def test_system_prompt_prefers_web_fetch_for_known_http_url(self):
        """验证已知网页地址会获得专用抓取工具而非 Bash 的使用指引。"""
        agent = Agent(
            model=object(),
            tools=[BashTool(), WebFetchTool()],
        )

        prompt = agent._build_system_prompt("读取 https://example.com/article")

        self.assertIn("use web_fetch first", prompt)
        self.assertIn("Do not use bash", prompt)


if __name__ == "__main__":
    unittest.main()
