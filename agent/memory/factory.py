# encoding:utf-8
"""
MemoryManager 工厂函数。
"""

import logging
from typing import Any

from agent.memory.config import MemoryConfig, get_embedding_func
from agent.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


def create_memory_manager(
    llm_model: Any = None,
    workspace_dir: str = "",
) -> MemoryManager:
    """
    根据显式工作区或全局配置创建 MemoryManager。
    """
    from config import conf

    provider = conf().get("embedding_provider", "ollama")
    if provider not in ("ollama", "openai"):
        logger.warning(
            "[MemoryFactory] embedding_provider=%s is configured, "
            "but only Ollama and OpenAI embedding are implemented for now",
            provider,
        )

    memory_config = MemoryConfig(
        workspace_dir=workspace_dir or conf().get("agent_workspace", "~/myagent"),
        embedding_api_base=conf().get(
            "embedding_api_base",
            "http://localhost:11434/api/embeddings",
        ),
        embedding_model=conf().get("embedding_model", "bge-m3"),
    )
    embedding_func = get_embedding_func(
        api_base=(
            conf().get("openai_api_base", "https://api.openai.com/v1")
            if provider == "openai"
            else memory_config.embedding_api_base
        ),
        model=(
            conf().get("embedding_openai_model", "text-embedding-3-small")
            if provider == "openai"
            else memory_config.embedding_model
        ),
        provider=provider,
        api_key=conf().get("openai_api_key", ""),
    )
    return MemoryManager(
        config=memory_config,
        embedding_func=embedding_func,
        llm_model=llm_model,
    )
