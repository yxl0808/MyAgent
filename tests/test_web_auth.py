import unittest

from channel.web.auth import (
    BrowserSessionManager,
    CsrfError,
    require_same_origin,
)


class BrowserSessionManagerTest(unittest.TestCase):
    def test_create_and_get_local_browser_sessions(self):
        """验证本地浏览器会话使用独立随机凭证并可按令牌恢复。"""
        manager = BrowserSessionManager()

        first = manager.create_session()
        second = manager.create_session()

        self.assertNotEqual(first.token, second.token)
        self.assertNotEqual(first.csrf_token, second.csrf_token)
        self.assertEqual(manager.get_session(first.token), first)
        self.assertEqual(manager.get_session(second.token), second)
        self.assertIsNone(manager.get_session(None))
        self.assertIsNone(manager.get_session("unknown"))

    def test_require_csrf_accepts_only_matching_token(self):
        """验证修改请求只能使用当前浏览器会话的 CSRF 令牌。"""
        manager = BrowserSessionManager()
        session = manager.create_session()

        manager.require_csrf(session, session.csrf_token)

        with self.assertRaises(CsrfError):
            manager.require_csrf(session, None)
        with self.assertRaises(CsrfError):
            manager.require_csrf(session, "wrong")

    def test_require_same_origin_rejects_missing_or_cross_origin_requests(self):
        """验证修改请求必须来自当前 Web Console 地址。"""
        require_same_origin("http://localhost:9899", "localhost:9899")
        require_same_origin("https://127.0.0.1:9899", "127.0.0.1:9899")

        invalid_requests = (
            (None, "localhost:9899"),
            ("http://localhost:9899", None),
            ("https://evil.example", "localhost:9899"),
            ("ftp://localhost:9899", "localhost:9899"),
        )
        for origin, host in invalid_requests:
            with self.subTest(origin=origin, host=host):
                with self.assertRaises(CsrfError):
                    require_same_origin(origin, host)


if __name__ == "__main__":
    unittest.main()
