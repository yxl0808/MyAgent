"""MyAgent Web Console 的 SQLite 持久化边界。"""

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from channel.web.schemas import MessageRecord, SessionRecord
from common.paths import resolve_path


class SessionNotFoundError(LookupError):
    """表示请求的 Web 会话不存在。"""


def _utc_now() -> str:
    """返回适合持久化的 ISO 8601 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    """将 SQLite 查询行转换为不可变的会话记录。"""
    return SessionRecord(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    """将持久化的消息数据解码为不可变消息记录。"""
    return MessageRecord(
        session_id=row["session_id"],
        sequence=row["sequence"],
        payload=json.loads(row["payload_json"]),
        created_at=row["created_at"],
    )


class WebConsoleStore:
    """管理 Web Console 的 SQLite 持久化操作。"""

    def __init__(self, db_path: str | Path):
        """保存数据库绝对路径并创建可重入写锁。"""
        self.db_path = resolve_path(db_path)
        self._write_lock = threading.RLock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """提供已配置的 SQLite 连接，并负责提交、回滚和关闭。"""
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """创建 Web 数据库目录和初始表结构。"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS web_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS web_messages (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY (session_id)
                        REFERENCES web_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_web_sessions_updated
                    ON web_sessions(updated_at DESC);
                """
            )

    def create_session(self, title: str = "新对话") -> SessionRecord:
        """创建并返回一个持久化的 Web 会话。"""
        session_id = str(uuid.uuid4())
        now = _utc_now()
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO web_sessions(id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return SessionRecord(session_id, title, now, now)

    def list_sessions(self) -> list[SessionRecord]:
        """按最近更新时间从新到旧返回 Web 会话。"""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM web_sessions ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        return [_row_to_session(row) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord:
        """返回指定 Web 会话，不存在时抛出专用异常。"""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM web_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return _row_to_session(row)

    def rename_session(self, session_id: str, title: str) -> SessionRecord:
        """更新 Web 会话标题和最后修改时间。"""
        normalized = title.strip()
        if not 1 <= len(normalized) <= 100:
            raise ValueError("会话标题长度必须为 1 到 100 个字符")
        now = _utc_now()
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE web_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (normalized, now, session_id),
            )
            if cursor.rowcount == 0:
                raise SessionNotFoundError(session_id)
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        """删除指定 Web 会话及其全部消息。"""
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM web_sessions WHERE id = ?",
                (session_id,),
            )
            if cursor.rowcount == 0:
                raise SessionNotFoundError(session_id)

    def append_message(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> MessageRecord:
        """为指定会话分配顺序编号并追加一条消息。"""
        now = _utc_now()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock, self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM web_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if exists is None:
                raise SessionNotFoundError(session_id)
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 "
                "FROM web_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO web_messages"
                "(session_id, sequence, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, sequence, encoded, now),
            )
            connection.execute(
                "UPDATE web_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
        return MessageRecord(session_id, sequence, dict(payload), now)

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        """按照稳定顺序返回指定会话的全部消息。"""
        self.get_session(session_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, sequence, payload_json, created_at "
                "FROM web_messages WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [_row_to_message(row) for row in rows]
