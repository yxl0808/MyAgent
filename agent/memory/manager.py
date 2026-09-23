# encoding:utf-8
"""
MemoryManager — 记忆系统统一入口。
混合搜索 = ChromaDB 向量 + SQLite FTS5 trigram。
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory.config import MemoryConfig
from agent.memory.storage import MemoryStorage, SearchResult
from agent.memory.chunker import TextChunker
from agent.memory.summarizer import MemoryFlushManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆系统统一入口。Agent 和工具只通过它访问记忆。"""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
        llm_model: Any = None,
    ):
        self.config = config or MemoryConfig()
        self.embedding_func = embedding_func
        self.llm_model = llm_model

        # SQLite 存储
        self.storage = MemoryStorage(self.config.db_path)

        # 分块器
        self.chunker = TextChunker(
            max_tokens=self.config.chunk_max_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
        )

        # ChromaDB
        self._chroma_client = None
        self._collection = None
        self._init_chroma()

        # 摘要蒸馏
        self.flush_manager = MemoryFlushManager(
            workspace_dir=self.config.get_workspace(),
            llm_model=self.llm_model,
        )

        if self.config.auto_sync:
            try:
                self.sync()
            except Exception as exc:
                logger.warning("[MemoryManager] Initial sync failed: %s", exc)

    def _get_auto_refine_state_file(self) -> Path:
        """Return the auto-refine state file path."""
        return Path(self.config.get_workspace()) / "memory" / ".auto-refine-state.json"

    def _load_auto_refine_state(self) -> Dict[str, Any]:
        """加载自动 refine 状态，读取失败时不影响 Agent 正常运行。"""
        state_file = self._get_auto_refine_state_file()
        try:
            raw_state = state_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            return {}
        except OSError:
            return {}
        if not raw_state.strip():
            return {}
        try:
            state = json.loads(raw_state)
        except json.JSONDecodeError:
            return {}
        return state if isinstance(state, dict) else {}

    def _save_auto_refine_state(self, state: Dict[str, Any]) -> bool:
        """使用原子文件替换持久化自动 refine 状态。"""
        state_file = self._get_auto_refine_state_file()
        temp_file = state_file.with_name(f"{state_file.name}.tmp")
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_file.replace(state_file)
            return True
        except (OSError, TypeError, ValueError):
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def _compute_daily_memory_hash(self, lookback_days: int = 7) -> str:
        """计算指定回溯天数内 daily memory 文件的稳定哈希。"""
        if lookback_days <= 0:
            return ''

        digest = hashlib.sha256()
        found_daily_memory = False
        memory_dir = self.config.get_memory_dir()
        today = datetime.now().date()

        for offset in range(lookback_days - 1, -1, -1):
            day = today - timedelta(days=offset)
            daily_file = memory_dir / f'{day:%Y-%m-%d}.md'
            try:
                content = daily_file.read_bytes()
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning('[MemoryManager] Failed to read daily memory: %s', exc)
                continue

            digest.update(daily_file.name.encode('utf-8'))
            digest.update(b'\0')
            digest.update(content)
            digest.update(b'\0')
            found_daily_memory = True

        return digest.hexdigest() if found_daily_memory else ''

    def _should_auto_refine(
        self,
        auto_refine_enabled: bool,
        lookback_days: int = 7,
        now: Optional[datetime] = None,
    ) -> bool:
        """判断当前是否满足自动 daily refine 的执行条件。"""
        if not auto_refine_enabled:
            return False

        current_time = now or datetime.now()
        if current_time.hour < 18:
            return False

        state = self._load_auto_refine_state()
        if state.get('last_auto_success_date') == current_time.date().isoformat():
            return False

        daily_hash = self._compute_daily_memory_hash(lookback_days=lookback_days)
        if not daily_hash:
            return False

        return state.get('last_processed_daily_hash') != daily_hash

    def _run_auto_refine(
        self,
        auto_refine_enabled: bool,
        lookback_days: int = 7,
        now: Optional[datetime] = None,
    ) -> bool:
        """执行一次满足条件的自动 refine，并在成功后保存处理状态。"""
        current_time = now or datetime.now()
        try:
            if not self._should_auto_refine(
                auto_refine_enabled,
                lookback_days=lookback_days,
                now=current_time,
            ):
                return False

            daily_hash = self._compute_daily_memory_hash(lookback_days=lookback_days)
            if not daily_hash or not self.refine(lookback_days=lookback_days):
                return False

            state = self._load_auto_refine_state()
            state['last_auto_success_date'] = current_time.date().isoformat()
            state['last_processed_daily_hash'] = daily_hash
            return self._save_auto_refine_state(state)
        except Exception as exc:
            logger.warning('[MemoryManager] Auto refine failed: %s', exc)
            return False

    def _init_chroma(self):
        """初始化 ChromaDB PersistentClient + collection。"""
        try:
            import chromadb
            d = self.config.get_chroma_dir()
            d.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(d))
            self._collection = self._chroma_client.get_or_create_collection(
                name="myagent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[MemoryManager] ChromaDB ready: %s", d)
        except ImportError:
            logger.warning("[MemoryManager] chromadb not installed")
        except Exception as e:
            logger.warning("[MemoryManager] ChromaDB init failed: %s", e)

    # ── 搜 索 ────────────────────────────────────────────────

    def search(self, query: str, limit: int = 0) -> List[SearchResult]:
        """混合搜索：向量 + 关键词 → 合并排序。"""
        if self.config.sync_on_search:
            try:
                self.sync()
            except Exception as exc:
                logger.warning('[MemoryManager] Pre-search sync failed: %s', exc)

        limit = limit or self.config.max_results

        # 1. 关键词搜索（始终可用）
        kw_results = self.storage.search_keyword(query, limit * 2)
        logger.info("[MemoryManager] Keyword: %d hits", len(kw_results))

        # 2. 向量搜索（需 embedding_func + chroma 可用）
        vec_results: List[SearchResult] = []
        if self.embedding_func and self._collection:
            try:
                qvec = self.embedding_func([query])[0]
                raw = self._collection.query(
                    query_embeddings=[qvec], n_results=limit * 2,
                )
                for i, mid in enumerate(raw["ids"][0]):
                    meta = raw["metadatas"][0][i]
                    dist = raw["distances"][0][i]
                    vec_results.append(SearchResult(
                        path=meta.get("path", ""),
                        start_line=meta.get("start_line", 0),
                        end_line=meta.get("end_line", 0),
                        score=1.0 - dist,  # cosine 距离→相似度
                        snippet=(raw["documents"][0][i] or "")[:500],
                        source=meta.get("source", "memory"),
                    ))
                logger.info("[MemoryManager] Vector: %d hits", len(vec_results))
            except Exception as e:
                logger.warning("[MemoryManager] Vector search failed: %s", e)

        # 3. 合并排序
        return self._merge(vec_results, kw_results,
                           self.config.vector_weight,
                           self.config.keyword_weight)[:limit]

    def _merge(
        self,
        vec: List[SearchResult], kw: List[SearchResult],
        vw: float, kw_w: float,
    ) -> List[SearchResult]:
        """加权合并两路结果，按 (path, start_line, end_line) 去重。"""
        if not vec:
            return sorted(kw, key=lambda r: r.score, reverse=True)
        if not kw:
            return sorted(vec, key=lambda r: r.score, reverse=True)

        merged: Dict[tuple, SearchResult] = {}
        for r in vec:
            k = (r.path, r.start_line, r.end_line)
            merged[k] = SearchResult(
                path=r.path, start_line=r.start_line, end_line=r.end_line,
                score=r.score * vw, snippet=r.snippet, source=r.source,
            )
        for r in kw:
            k = (r.path, r.start_line, r.end_line)
            if k in merged:
                merged[k].score += r.score * kw_w
            else:
                merged[k] = SearchResult(
                    path=r.path, start_line=r.start_line, end_line=r.end_line,
                    score=r.score * kw_w, snippet=r.snippet, source=r.source,
                )
        return sorted(merged.values(), key=lambda r: r.score, reverse=True)

    # ── 查 询 ────────────────────────────────────────────────

    def get_recent(self, limit: int = 5) -> List[str]:
        """获取最近 N 条记忆文本。"""
        return self.storage.get_recent(limit)

    # ── 写 入 ────────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        source: str = "memory",
        importance: float = 0.5,
        path: str = "",
    ):
        """写入一条记忆：分块 → embedding → ChromaDB + SQLite。"""
        if not content.strip():
            return

        if not path:
            h = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
            path = f"memory/manual_{h}.md"

        chunks = self.chunker.chunk_text(content)
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = None
        if self.embedding_func:
            try:
                embeddings = self.embedding_func(texts)
            except Exception as e:
                logger.warning("[MemoryManager] Embedding failed: %s", e)

        sql_chunks = []
        for chunk in chunks:
            chunk_id = hashlib.md5(
                f"{path}:{chunk.start_line}:{chunk.end_line}".encode("utf-8")
            ).hexdigest()
            sql_chunks.append({
                "id": chunk_id,
                "path": path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "hash": MemoryStorage.compute_hash(chunk.text),
                "source": source,
                "importance": importance,
            })

        self.storage.save_chunks_batch(sql_chunks)

        if embeddings and self._collection:
            try:
                self._collection.add(
                    ids=[c["id"] for c in sql_chunks],
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=[{
                        "path": c["path"],
                        "start_line": c["start_line"],
                        "end_line": c["end_line"],
                        "source": c["source"],
                        "importance": c["importance"],
                    } for c in sql_chunks],
                )
            except Exception as e:
                logger.warning("[MemoryManager] ChromaDB add failed: %s", e)

        logger.info("[MemoryManager] Added %d chunks: %s", len(sql_chunks), path)

    # ── 同 步 ────────────────────────────────────────────────

    def sync(self, force: bool = False):
        """增量扫描 MEMORY.md 和 memory/*.md，变化文件重新索引。"""
        workspace = self.config.get_workspace()
        memory_dir = self.config.get_memory_dir()

        files_to_scan = []

        main_memory = workspace / "MEMORY.md"
        if main_memory.exists():
            files_to_scan.append((main_memory, "memory"))

        if memory_dir.exists():
            for fpath in memory_dir.rglob("*.md"):
                rel_parts = fpath.relative_to(memory_dir).parts
                if any(part.startswith(".") for part in rel_parts):
                    continue
                files_to_scan.append((fpath, "memory"))

        if not files_to_scan:
            logger.info("[MemoryManager] Sync skipped: no memory files")
            return

        pending = []
        for fpath, source in files_to_scan:
            try:
                content = fpath.read_text(encoding="utf-8")
                rel_path = str(fpath.relative_to(workspace)).replace("\\", "/")
                stat = fpath.stat()
            except Exception as e:
                logger.warning("[MemoryManager] Skip unreadable memory file %s: %s", fpath, e)
                continue

            file_hash = MemoryStorage.compute_hash(content)
            if not force and self.storage.get_file_hash(rel_path) == file_hash:
                continue

            chunks = self.chunker.chunk_text(content)
            pending.append({
                "rel_path": rel_path,
                "file_hash": file_hash,
                "source": source,
                "chunks": chunks,
                "texts": [chunk.text for chunk in chunks],
                "stat": stat,
            })

        if not pending:
            logger.info("[MemoryManager] Sync skipped: no changed files")
            return

        all_texts = []
        for entry in pending:
            all_texts.extend(entry["texts"])

        all_embeddings = None
        if all_texts and self.embedding_func:
            try:
                all_embeddings = self.embedding_func(all_texts)
                if len(all_embeddings) != len(all_texts):
                    logger.warning(
                        "[MemoryManager] Embedding count mismatch: %d texts, %d vectors",
                        len(all_texts), len(all_embeddings),
                    )
                    all_embeddings = None
            except Exception as e:
                logger.warning("[MemoryManager] Batch embedding failed: %s", e)

        embedding_index = 0
        total_chunks = 0

        for entry in pending:
            rel_path = entry["rel_path"]
            texts = entry["texts"]
            chunks = entry["chunks"]
            n = len(texts)

            self.storage.delete_by_path(rel_path)
            if self._collection:
                try:
                    self._collection.delete(where={"path": rel_path})
                except Exception as e:
                    logger.warning("[MemoryManager] ChromaDB delete failed for %s: %s", rel_path, e)

            entry_embeddings = None
            if all_embeddings is not None:
                entry_embeddings = all_embeddings[embedding_index:embedding_index + n]
            embedding_index += n

            if not chunks:
                stat = entry["stat"]
                self.storage.update_file_metadata(
                    rel_path,
                    entry["file_hash"],
                    int(stat.st_mtime),
                    stat.st_size,
                )
                continue

            sql_chunks = []
            for chunk in chunks:
                chunk_id = hashlib.md5(
                    f"{rel_path}:{chunk.start_line}:{chunk.end_line}".encode("utf-8")
                ).hexdigest()
                sql_chunks.append({
                    "id": chunk_id,
                    "path": rel_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                    "hash": MemoryStorage.compute_hash(chunk.text),
                    "source": entry["source"],
                    "importance": 0.5,
                })

            self.storage.save_chunks_batch(sql_chunks)
            total_chunks += len(sql_chunks)

            if entry_embeddings is not None and self._collection:
                try:
                    self._collection.add(
                        ids=[chunk["id"] for chunk in sql_chunks],
                        documents=texts,
                        embeddings=entry_embeddings,
                        metadatas=[{
                            "path": chunk["path"],
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "source": chunk["source"],
                            "importance": chunk["importance"],
                        } for chunk in sql_chunks],
                    )
                except Exception as e:
                    logger.warning("[MemoryManager] ChromaDB sync add failed for %s: %s", rel_path, e)

            stat = entry["stat"]
            self.storage.update_file_metadata(
                rel_path,
                entry["file_hash"],
                int(stat.st_mtime),
                stat.st_size,
            )

        logger.info(
            "[MemoryManager] Sync complete: %d files, %d chunks",
            len(pending),
            total_chunks,
        )

    def rebuild_vectors(self) -> int:
        """使用 SQLite 中的所有文本块重建 ChromaDB 向量索引。"""
        if not self.embedding_func:
            logger.warning("[MemoryManager] Cannot rebuild vectors: embedding unavailable")
            return 0

        if not self._chroma_client:
            logger.warning("[MemoryManager] Cannot rebuild vectors: ChromaDB unavailable")
            return 0

        chunks = self.storage.get_all_chunks()

        try:
            self._chroma_client.delete_collection("myagent_memory")
        except Exception:
            pass

        self._collection = self._chroma_client.get_or_create_collection(
            name="myagent_memory",
            metadata={"hnsw:space": "cosine"},
        )

        if not chunks:
            logger.info("[MemoryManager] Rebuilt vectors: 0 chunks")
            return 0

        texts = [chunk["text"] for chunk in chunks]
        try:
            embeddings = self.embedding_func(texts)
        except Exception as e:
            logger.warning("[MemoryManager] Rebuild embedding failed: %s", e)
            return 0

        if len(embeddings) != len(chunks):
            logger.warning(
                "[MemoryManager] Rebuild embedding count mismatch: %d chunks, %d vectors",
                len(chunks),
                len(embeddings),
            )
            return 0

        self._collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "path": chunk["path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "source": chunk["source"],
                "importance": chunk["importance"],
            } for chunk in chunks],
        )
        logger.info("[MemoryManager] Rebuilt vectors: %d chunks", len(chunks))
        return len(chunks)

    def flush(self, messages: List[Dict], reason: str = "threshold") -> bool:
        """对话摘要写入 daily memory，并同步进长期记忆索引。"""
        if not self.flush_manager:
            return False

        ok = self.flush_manager.flush_from_messages(messages, reason=reason)
        if ok:
            self.sync()
        return ok

    def refine(self, lookback_days: int = 7, force: bool = False) -> bool:
        """蒸馏 daily memory 到 MEMORY.md，并同步进长期记忆索引。"""
        if not self.flush_manager:
            return False

        ok = self.flush_manager.refine(
            lookback_days=lookback_days,
            force=force,
        )
        if ok:
            self.sync()
        return ok

    def get_status(self) -> Dict[str, Any]:
        """返回记忆系统当前状态，供工具和调试界面使用。"""
        stats = self.storage.get_stats()
        return {
            "chunks": stats["chunks"],
            "files": stats["files"],
            "workspace": str(self.config.get_workspace()),
            "chroma_available": self._collection is not None,
            "embedding_available": self._is_embedding_available(),
            "embedding_model": self.config.embedding_model,
        }

    def _is_embedding_available(self) -> bool:
        """实际探测 embedding 函数是否能生成向量。"""
        if not self.embedding_func:
            return False

        try:
            embeddings = self.embedding_func(["ping"])
        except Exception as e:
            logger.warning("[MemoryManager] Embedding availability check failed: %s", e)
            return False

        return bool(embeddings and embeddings[0])

    def close(self):
        """关闭底层存储连接。"""
        self.storage.close()
