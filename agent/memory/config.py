# encoding:utf-8
"""
记忆系统配置 — MemoryConfig + EmbeddingProvider
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """
    记忆系统所有配置。有默认值，可逐项覆盖。

    workspace_dir         — 工作区根目录（空则读全局 config）
    chroma_persist_dir    — ChromaDB 持久化目录
    db_path               — SQLite 索引路径
    chunk_max_tokens      — 每块最大 token 数，默认 500
    chunk_overlap_tokens  — 相邻块重叠 token 数，默认 50
    max_results           — 搜索最多返回条数，默认 10
    min_score             — 最低分数阈值，默认 0.1
    vector_weight         — 向量搜索权重，默认 0.7
    keyword_weight        — 关键词搜索权重，默认 0.3
    embedding_api_base    — Ollama API 地址
    embedding_model       — embedding 模型名，默认 bge-m3
    auto_sync             — 启动时自动扫描文件
    sync_on_search        — 搜索前先同步
    """

    workspace_dir: str = ""
    chroma_persist_dir: str = ""
    db_path: str = ""

    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    max_results: int = 10
    min_score: float = 0.1
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

    embedding_api_base: str = "http://localhost:11434/api/embeddings"
    embedding_model: str = "bge-m3"

    auto_sync: bool = True
    sync_on_search: bool = True

    def __post_init__(self):
        """自动推导未设置的路径。"""
        if not self.workspace_dir:
            from config import conf
            self.workspace_dir = conf().get("agent_workspace", "~/myagent")
        ws = Path(self.workspace_dir).expanduser().resolve()
        if not self.chroma_persist_dir:
            self.chroma_persist_dir = str(ws / "chroma")
        if not self.db_path:
            index_dir = ws / "memory" / "long-term"
            index_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(index_dir / "index.db")

    def get_workspace(self) -> Path:
        """返回 workspace 根目录绝对路径。"""
        return Path(self.workspace_dir).expanduser().resolve()

    def get_memory_dir(self) -> Path:
        """返回 memory/ 目录路径。"""
        return self.get_workspace() / "memory"

    def get_chroma_dir(self) -> Path:
        """返回 ChromaDB 持久化目录。"""
        return Path(self.chroma_persist_dir).expanduser().resolve()


class EmbeddingProvider:
    """   
    调用 Ollama embedding API。

    接口:
        embed_batch(texts: List[str]) -> List[List[float]]
        embed_query(text: str) -> List[float]

    用法:
        provider = EmbeddingProvider(model="bge-m3")
        vectors = provider.embed_batch(["你好", "hello"])
    """

    def __init__(self, api_base: str = "", model: str = ""):
        import requests as _requests
        self._requests = _requests
        self.api_base = api_base or "http://localhost:11434/api/embeddings"
        self.model = model or "bge-m3"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """逐条调 Ollama API，返回等长向量列表。"""
        embeddings = []
        for text in texts:
            resp = self._requests.post(
                self.api_base,
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条查询转向量。"""
        return self.embed_batch([text])[0]


def get_embedding_func(
    api_base: str = "",
    model: str = "",
    provider: str = "ollama",
    api_key: str = "",
) -> Callable[[List[str]], List[List[float]]]:
    """
    返回 embed_batch 可调用对象。

    MemoryManager 通过此函数获取 embedding 能力，只需知道签名是
    func(texts) -> embeddings，不关心底层是 Ollama 还是 OpenAI。
    """
    if provider == "openai":
        import requests as _requests

        openai_api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        openai_model = model or "text-embedding-3-small"

        def embed_openai_batch(texts: List[str]) -> List[List[float]]:
            """批量调用 OpenAI embedding API，返回等长向量列表。"""
            resp = _requests.post(
                f"{openai_api_base}/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": openai_model, "input": texts},
                timeout=30,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in data]

        return embed_openai_batch

    if provider != "ollama":
        logger.warning(
            "[MemoryConfig] embedding provider %s is not implemented, fallback to ollama",
            provider,
        )

    embedding_provider = EmbeddingProvider(api_base=api_base, model=model)
    return embedding_provider.embed_batch
