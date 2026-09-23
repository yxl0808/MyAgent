# encoding:utf-8
"""
SQLite 存储层 — 关键词搜索 + 文本管理 + 文件元数据。
FTS5 trigram tokenizer 一套搞定中英文混合查询。
向量全在 ChromaDB，这里不管。
"""

import hashlib
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 模块级编译一次
_RE_CJK = re.compile(
    r"[　-ヿ㐀-鿿가-힯豈-﫿\U00020000-\U0002fa1f]"
)
_RE_TRIGRAM = re.compile(
    r"[　-ヿ㐀-鿿가-힯豈-﫿\U00020000-\U0002fa1f]+|[A-Za-z0-9_]+"
)


@dataclass
class SearchResult:
    """关键词搜索结果"""
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    source: str = "memory"
    #source — 这条记忆从哪来的，默认 "memory"。后续可扩展为 "conversation"（对话摘要）、"skill"（技能记录），搜索时可按来源过滤


class MemoryStorage:
    """SQLite 存储，管文本和关键词。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._init_db()

    # ── 建 表 ────────────────────────────────────────────────

    def _init_db(self):
        """建立数据库连接，自愈，建表。"""
        Path(self.db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
#默认行为：SQLite 默认禁止多个线程使用同一个数据库连接（会报错 SQLite objects created in a thread can only be used in that same thread）。
#
# 设置为 False：解除这个限制，允许多个线程共享这个连接对象。
#
# 为什么这么写？ 你的 Agent 服务通常是多线程运行（比如 FastAPI/Flask 处理并发请求）。如果不加这个参数，每次查询都要创建新连接，性能极差。设置为 False 配合后面的 WAL 模式，能安全地支持多线程并发读写
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")#Write-Ahead Logging，预写日志
        self.conn.execute("PRAGMA busy_timeout=5000")

        # 完整性自愈：数据库损坏则重建
        try:
            ok = self.conn.execute("PRAGMA integrity_check").fetchone()
            if ok[0] != "ok":
                raise sqlite3.DatabaseError(str(ok[0]))
        except sqlite3.DatabaseError:
            logger.warning("[MemoryStorage] DB corrupt, recreating")
            self.conn.close()
            Path(self.db_path).unlink(missing_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")

        # chunks 表 — 文本块
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                id TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                hash TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'memory',
                importance REAL NOT NULL DEFAULT 0.0,
                created_at INTEGER DEFAULT (strftime('%s','now')),
                updated_at INTEGER DEFAULT (strftime('%s','now'))
            )
        """)

        # files 表 — 文件元数据（用于增量 sync）
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s','now'))
            )
        """)

        # _meta 表 — 内部状态标记（如 trigram 回填完成）
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        self._init_fts5()
        self.conn.commit()

    def _init_fts5(self):
        """创建 FTS5 trigram 虚拟表 + 3 个同步触发器。状态不一致时自愈。"""

        # 探测 FTS5 是否可用
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _ft5p USING fts5(x)"
            )
            self.conn.execute("DROP TABLE IF EXISTS _ft5p")
        except sqlite3.OperationalError:
            logger.warning("[Storage] FTS5 unavailable, keyword search disabled")
            return

        # 自愈：表和触发器必须同时存在，否则全部重建
        has_table = self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='chunks_fts'"
        ).fetchone()
        trigs = self.conn.execute(
            "SELECT COUNT(*) as c FROM sqlite_master "
            "WHERE type='trigger' AND name IN "
            "('cfts_ai','cfts_ad','cfts_au')"
        ).fetchone()["c"]

        if bool(has_table) != (trigs > 0):
            logger.warning("[Storage] FTS5 inconsistent, resetting")
            for t in ("cfts_ai", "cfts_ad", "cfts_au"):
                self.conn.execute(f"DROP TRIGGER IF EXISTS {t}")
            self.conn.execute("DROP TABLE IF EXISTS chunks_fts")

        # trigram FTS5 虚拟表
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                id UNINDEXED,
                path UNINDEXED,
                source UNINDEXED,
                content='chunks',
                content_rowid='rowid',
                tokenize='trigram case_sensitive 0'
            )
        """)

        # 三个触发器：INSERT / DELETE / UPDATE chunks → 自动同步 chunks_fts
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS cfts_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text, id, path, source)
                VALUES (new.rowid, new.text, new.id, new.path, new.source);
            END
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS cfts_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.rowid;
            END
        """)
        self.conn.execute("""
            CREATE TRIGGER IF NOT EXISTS cfts_au AFTER UPDATE ON chunks BEGIN
                UPDATE chunks_fts SET
                    text=new.text, id=new.id, path=new.path, source=new.source
                WHERE rowid = new.rowid;
            END
        """)

    # ── 写 入 ────────────────────────────────────────────────

    def save_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """批量写入 chunk，INSERT 后 FTS5 触发器自动建索引。"""
        if not chunks:
            return

        _USE_UPSERT = sqlite3.sqlite_version_info >= (3, 24, 0)
        if _USE_UPSERT:
            sql = """
                INSERT INTO chunks (id, path, start_line, end_line, text,
                                    hash, source, importance, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text, hash=excluded.hash,
                    source=excluded.source, importance=excluded.importance,
                    updated_at=strftime('%s','now')
            """
        else:
            sql = """
                INSERT OR REPLACE INTO chunks (id, path, start_line, end_line,
                    text, hash, source, importance, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))
            """
        rows = [
            (c["id"], c["path"], c["start_line"], c["end_line"],
             c["text"], c["hash"], c.get("source", "memory"),
             c.get("importance", 0.0))
            for c in chunks
        ]
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    # ── 搜 索 ────────────────────────────────────────────────

    def search_keyword(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        FTS5 trigram 关键词搜索。FTS5 不可用时回退 LIKE。
        中英文混合查询自动适配。
        """
        if not query.strip():
            return []

        # 构建 trigram MATCH 查询
        tokens = [t for t in _RE_TRIGRAM.findall(query) if t]
        if tokens:
            quoted = [f'"{t.replace(chr(34), chr(34)*2)}"' for t in tokens]
            fts_query = " AND ".join(quoted)
            try:
                rows = self.conn.execute("""
                    SELECT chunks.*, bm25(chunks_fts) as rank
                    FROM chunks_fts
                    JOIN chunks ON chunks.rowid = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, limit)).fetchall()
                if rows:
                    return [
                        SearchResult(
                            path=r["path"], start_line=r["start_line"],
                            end_line=r["end_line"],
                            score=0.3 + 0.69 * abs(r["rank"]) / (1.0 + abs(r["rank"])),
                            snippet=r["text"][:500] if len(r["text"]) > 500 else r["text"],
                            source=r["source"],
                        )
                        for r in rows
                    ]
            except Exception as e:
                logger.warning("[Storage] FTS5 search failed: %s, trying LIKE", e)

        # LIKE 回退（CJK 查询或 FTS5 失败时）
        return self._search_like(query, limit)

    def _search_like(self, query: str, limit: int) -> List[SearchResult]:
        """LIKE 回退搜索。CJK 连续字符 + ASCII 3+ 字符分词匹配。"""
        cjk_words = re.findall(
            r"[　-ヿ㐀-鿿가-힯豈-﫿\U00020000-\U0002fa1f]+", query
        )
        ascii_words = [w for w in re.findall(r"[A-Za-z0-9_]+", query) if len(w) >= 3]
        words = cjk_words + ascii_words
        if not words:
            return []

        conditions = " OR ".join(["LOWER(text) LIKE ?"] * len(words))
        params = [f"%{w.lower()}%" for w in words] + [limit]

        try:
            rows = self.conn.execute(
                f"SELECT * FROM chunks WHERE ({conditions}) LIMIT ?", params
            ).fetchall()
        except Exception as e:
            logger.warning("[Storage] LIKE search failed: %s", e)
            return []

        results = []
        for r in rows:
            tl = r["text"].lower()
            matched = sum(1 for w in words if w.lower() in tl)
            if matched == 0:
                continue
            results.append(SearchResult(
                path=r["path"], start_line=r["start_line"],
                end_line=r["end_line"],
                score=min(0.85, 0.3 + 0.15 * matched),
                snippet=r["text"][:500] if len(r["text"]) > 500 else r["text"],
                source=r["source"],
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # ── 文件元数据 ─────────────────────────────────────────────

    def get_file_hash(self, path: str) -> Optional[str]:
        """取已存储的文件 hash。返回 None 表示未索引过或内容已变。"""
        row = self.conn.execute(
            "SELECT hash FROM files WHERE path=?", (path,)
        ).fetchone()
        return row["hash"] if row else None

    def update_file_metadata(self, path: str, file_hash: str, mtime: int, size: int):
        """记录文件的 hash/mtime/size，sync 时判断是否需要重新索引。"""
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, hash, mtime, size, updated_at) "
                "VALUES (?, ?, ?, ?, strftime('%s','now'))",
                (path, file_hash, mtime, size),
            )
            self.conn.commit()

    def delete_by_path(self, path: str):
        """删除一个文件的所有 chunks + 元数据。sync 时先删后写。"""
        with self._lock:
            self.conn.execute("DELETE FROM chunks WHERE path=?", (path,))
            self.conn.execute("DELETE FROM files WHERE path=?", (path,))
            self.conn.commit()

    @staticmethod
    def compute_hash(content: str) -> str:
        """计算文本内容指纹，用于文件增量同步和 chunk 去重。"""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ── 查 询 ────────────────────────────────────────────────

    def get_recent(self, limit: int = 5) -> List[str]:
        """取最近 N 条记忆文本。Agent 注入 system prompt 用。"""
        rows = self.conn.execute(
            "SELECT text FROM chunks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["text"] for r in rows]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """返回所有文本块，用于重建向量索引。"""
        rows = self.conn.execute(
            "SELECT id, path, start_line, end_line, text, source, importance "
            "FROM chunks ORDER BY path ASC, start_line ASC, end_line ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, int]:
        """返回 chunks + files 统计。"""
        c = self.conn.execute("SELECT COUNT(*) as c FROM chunks").fetchone()["c"]
        f = self.conn.execute("SELECT COUNT(*) as c FROM files").fetchone()["c"]
        return {"chunks": c, "files": f}

    def close(self):
        """关闭数据库连接。"""
        if self.conn:
            self.conn.commit()
            self.conn.close()
            self.conn = None
