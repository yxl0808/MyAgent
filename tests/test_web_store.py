import sqlite3
import tempfile
import unittest
from pathlib import Path

import channel.web.store as store_module
from channel.web.schemas import SessionRecord, derive_session_title
from channel.web.store import SessionNotFoundError, WebConsoleStore


class WebConsoleStoreTest(unittest.TestCase):
    def setUp(self):
        """为每项测试创建隔离并已初始化的 Web 数据库。"""
        self.temp_dir = tempfile.TemporaryDirectory(prefix="myagent_web_store_")
        self.db_path = Path(self.temp_dir.name) / "web-console.db"
        self.store = WebConsoleStore(self.db_path)
        self.store.initialize()

    def tearDown(self):
        """释放临时 Web 数据库目录。"""
        self.temp_dir.cleanup()

    def test_row_to_session_converts_sqlite_row(self):
        """验证数据库行会转换为不可变的会话记录。"""
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """
                SELECT
                    'session-1' AS id,
                    '新会话' AS title,
                    '2026-07-30T10:00:00+00:00' AS created_at,
                    '2026-07-30T11:00:00+00:00' AS updated_at
                """
            ).fetchone()
        finally:
            connection.close()

        session = store_module._row_to_session(row)

        self.assertEqual(
            session,
            SessionRecord(
                id="session-1",
                title="新会话",
                created_at="2026-07-30T10:00:00+00:00",
                updated_at="2026-07-30T11:00:00+00:00",
            ),
        )

    def test_initialize_creates_web_console_schema(self):
        """验证初始化会创建 Web Console 所需的数据库结构。"""
        with tempfile.TemporaryDirectory(prefix="myagent_web_store_test_") as temp_dir:
            db_path = Path(temp_dir) / "web" / "web-console.db"
            store = WebConsoleStore(db_path)

            store.initialize()

            connection = sqlite3.connect(db_path)
            try:
                objects = dict(
                    connection.execute(
                        """
                        SELECT name, type
                        FROM sqlite_master
                        WHERE name IN (
                            'web_sessions',
                            'web_messages',
                            'idx_web_sessions_updated'
                        )
                        """
                    ).fetchall()
                )
            finally:
                connection.close()

            self.assertEqual(objects["web_sessions"], "table")
            self.assertEqual(objects["web_messages"], "table")
            self.assertEqual(objects["idx_web_sessions_updated"], "index")

    def test_session_crud_and_newest_first_order(self):
        """验证会话创建、排序、重命名和删除入口。"""
        first = self.store.create_session()
        second = self.store.create_session("第二个会话")

        renamed = self.store.rename_session(first.id, " 设计讨论 ")

        self.assertEqual(first.title, "新对话")
        self.assertEqual(renamed.title, "设计讨论")
        self.assertEqual(
            [item.id for item in self.store.list_sessions()],
            [first.id, second.id],
        )
        self.store.delete_session(first.id)
        with self.assertRaises(SessionNotFoundError):
            self.store.get_session(first.id)

    def test_messages_persist_across_store_instances(self):
        """验证消息顺序、中文内容和上下文标记跨实例保存。"""
        session = self.store.create_session()
        self.store.append_message(
            session.id,
            {
                "kind": "user",
                "content": "你好",
                "include_in_context": True,
                "context_messages": [{"role": "user", "content": "你好"}],
            },
        )
        self.store.append_message(
            session.id,
            {
                "kind": "assistant",
                "content": "已停止",
                "include_in_context": False,
                "context_messages": [],
            },
        )

        reopened = WebConsoleStore(self.db_path)
        reopened.initialize()
        messages = reopened.list_messages(session.id)

        self.assertEqual([item.sequence for item in messages], [1, 2])
        self.assertEqual(messages[0].payload["content"], "你好")
        self.assertIs(messages[0].payload["include_in_context"], True)
        self.assertIs(messages[1].payload["include_in_context"], False)

    def test_delete_session_cascades_messages(self):
        """验证删除会话时 SQLite 外键级联删除消息。"""
        session = self.store.create_session()
        self.store.append_message(session.id, {"kind": "user"})

        self.store.delete_session(session.id)

        connection = sqlite3.connect(self.db_path)
        try:
            count = connection.execute(
                "SELECT COUNT(*) FROM web_messages WHERE session_id = ?",
                (session.id,),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 0)

    def test_derive_session_title_normalizes_and_limits_message(self):
        """验证首条消息会生成规范化且不超过限制的标题。"""
        self.assertEqual(derive_session_title("  项目讨论  "), "项目讨论")
        self.assertEqual(derive_session_title("   "), "新对话")
        self.assertEqual(derive_session_title("一" * 31), "一" * 30)


if __name__ == "__main__":
    unittest.main()
