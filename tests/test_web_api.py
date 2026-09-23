import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from channel.web.__main__ import validate_web_settings
from channel.web.app import create_web_app


class EventAgent:
    """生成文本、思考和工具事件的确定性 Web API 测试 Agent。"""

    def __init__(self):
        """创建符合 Agent 协作接口的消息历史。"""
        self.messages = []

    def run(self, user_message, stream=False, cancel_event=None):
        """输出完整的流式事件序列并保存最终回答。"""
        self.messages.append({"role": "user", "content": user_message})
        yield {"type": "thinking", "content": "正在分析", "step": 1}
        yield {"type": "tool_start", "name": "lookup", "args": {"q": user_message}}
        yield {
            "type": "tool_end",
            "name": "lookup",
            "output": "found",
            "success": True,
        }
        answer = f"回答：{user_message}"
        yield {"type": "text", "content": answer, "step": 1}
        self.messages.append({"role": "assistant", "content": answer})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer=answer, success=True),
        }


class BlockingAgent:
    """等待取消信号的确定性 Web API 测试 Agent。"""

    def __init__(self):
        """创建启动通知事件和消息历史。"""
        self.messages = []
        self.started = threading.Event()

    def run(self, user_message, stream=False, cancel_event=None):
        """保持运行状态，直到 RunManager 请求协作停止。"""
        self.messages.append({"role": "user", "content": user_message})
        self.started.set()
        cancel_event.wait(2.0)
        if cancel_event.is_set():
            yield {"type": "stopped"}
            return
        self.messages.append({"role": "assistant", "content": "超时完成"})
        yield {
            "type": "done",
            "result": SimpleNamespace(final_answer="超时完成", success=True),
        }


class WebApiTest(unittest.TestCase):
    def setUp(self):
        """为每项 API 测试启动隔离 FastAPI 应用和临时数据库。"""
        self.temp_dir = tempfile.TemporaryDirectory(prefix="myagent_web_api_")
        self.agent = EventAgent()
        self.app = create_web_app(
            {"agent_workspace": self.temp_dir.name, "web_show_thinking": False},
            agent=self.agent,
            db_path=Path(self.temp_dir.name) / "web-console.db",
        )
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        """关闭测试应用生命周期并清理临时目录。"""
        self.client.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def _headers(self, csrf_token):
        """构造通过同源和 CSRF 校验的修改请求头。"""
        return {"origin": "http://testserver", "x-csrf-token": csrf_token}

    def _initialize_browser(self):
        """访问首页并获取当前本机浏览器会话的 CSRF 令牌。"""
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        status = self.client.get("/api/auth/status")
        self.assertEqual(status.status_code, 200)
        return status.json()["csrf_token"]

    def _create_session(self, csrf_token):
        """通过受保护 API 创建一个空 Web 会话。"""
        response = self.client.post("/api/sessions", headers=self._headers(csrf_token))
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_static_page_and_local_browser_cookie(self):
        """验证首页、静态资源、本机会话 Cookie 和无外部 CDN 页面。"""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("myagent_web_session=", response.headers["set-cookie"])
        self.assertIn("HttpOnly", response.headers["set-cookie"])
        self.assertIn("SameSite=strict", response.headers["set-cookie"])
        self.assertNotIn("http://", response.text)
        self.assertNotIn("https://", response.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/styles.css").status_code, 200)

    def test_cookie_csrf_and_session_crud(self):
        """验证保护边界、会话 CRUD 与公开消息字段过滤。"""
        self.assertEqual(self.client.get("/api/sessions").status_code, 403)
        csrf_token = self._initialize_browser()
        self.assertEqual(self.client.post("/api/sessions").status_code, 403)
        session = self._create_session(csrf_token)

        renamed = self.client.patch(
            f"/api/sessions/{session['id']}",
            headers=self._headers(csrf_token),
            json={"title": "学习计划"},
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "学习计划")
        listed = self.client.get("/api/sessions")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["sessions"][0]["id"], session["id"])

        run = self.client.post(
            f"/api/sessions/{session['id']}/runs",
            headers=self._headers(csrf_token),
            json={"message": "隐藏内部上下文"},
        )
        self.assertEqual(run.status_code, 202)
        events = self.client.get(f"/api/runs/{run.json()['run_id']}/events")
        self.assertEqual(events.status_code, 200)
        messages = self.client.get(f"/api/sessions/{session['id']}/messages")
        self.assertEqual(messages.status_code, 200)
        self.assertNotIn("context_messages", messages.json()["messages"][-1])

        deleted = self.client.delete(
            f"/api/sessions/{session['id']}",
            headers=self._headers(csrf_token),
        )
        self.assertEqual(deleted.status_code, 204)

    def test_sse_event_replay_and_terminal_event(self):
        """验证 SSE 递增编号、终态事件和 Last-Event-ID 重放边界。"""
        csrf_token = self._initialize_browser()
        session = self._create_session(csrf_token)
        run = self.client.post(
            f"/api/sessions/{session['id']}/runs",
            headers=self._headers(csrf_token),
            json={"message": "测试 SSE"},
        ).json()

        response = self.client.get(f"/api/runs/{run['run_id']}/events")
        self.assertEqual(response.status_code, 200)
        event_ids = [
            int(line.removeprefix("id: "))
            for line in response.text.splitlines()
            if line.startswith("id: ")
        ]
        self.assertEqual(event_ids, list(range(1, len(event_ids) + 1)))
        self.assertIn("event: thinking", response.text)
        self.assertIn("event: tool_start", response.text)
        self.assertIn("event: tool_end", response.text)
        self.assertIn("event: text", response.text)
        self.assertIn("event: done", response.text)

        replay = self.client.get(
            f"/api/runs/{run['run_id']}/events",
            headers={"Last-Event-ID": "2"},
        )
        replay_ids = [
            int(line.removeprefix("id: "))
            for line in replay.text.splitlines()
            if line.startswith("id: ")
        ]
        self.assertTrue(replay_ids)
        self.assertTrue(all(event_id > 2 for event_id in replay_ids))

    def test_busy_stop_and_running_session_deletion(self):
        """验证共享 Agent 忙碌冲突、停止幂等和运行中会话保护。"""
        self.client.__exit__(None, None, None)
        self.agent = BlockingAgent()
        self.app = create_web_app(
            {"agent_workspace": self.temp_dir.name, "web_show_thinking": False},
            agent=self.agent,
            db_path=Path(self.temp_dir.name) / "blocking-web-console.db",
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        csrf_token = self._initialize_browser()
        first_session = self._create_session(csrf_token)
        second_session = self._create_session(csrf_token)
        run = self.client.post(
            f"/api/sessions/{first_session['id']}/runs",
            headers=self._headers(csrf_token),
            json={"message": "保持运行"},
        ).json()
        self.assertTrue(self.agent.started.wait(1.0))

        busy = self.client.post(
            f"/api/sessions/{second_session['id']}/runs",
            headers=self._headers(csrf_token),
            json={"message": "第二个运行"},
        )
        self.assertEqual(busy.status_code, 409)
        deleting = self.client.delete(
            f"/api/sessions/{first_session['id']}",
            headers=self._headers(csrf_token),
        )
        self.assertEqual(deleting.status_code, 409)

        first_stop = self.client.post(f"/api/runs/{run['run_id']}/stop", headers=self._headers(csrf_token))
        second_stop = self.client.post(f"/api/runs/{run['run_id']}/stop", headers=self._headers(csrf_token))
        self.assertEqual(first_stop.status_code, 200)
        self.assertEqual(second_stop.status_code, 200)
        events = self.client.get(f"/api/runs/{run['run_id']}/events")
        self.assertIn("event: stopped", events.text)

    def test_local_only_startup_validation(self):
        """验证入口拒绝非回环监听、禁用状态与非法端口。"""
        self.assertEqual(
            validate_web_settings({"web_enabled": True, "web_host": "localhost", "web_port": 9899}),
            ("localhost", 9899),
        )
        for values in (
            {"web_enabled": False},
            {"web_enabled": True, "web_host": "0.0.0.0", "web_port": 9899},
            {"web_enabled": True, "web_host": "127.0.0.1", "web_port": 0},
        ):
            with self.assertRaises(ValueError):
                validate_web_settings(values)


if __name__ == "__main__":
    unittest.main()
