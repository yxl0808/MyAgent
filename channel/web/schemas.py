"""MyAgent Web Console 共享数据类型。"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class SessionRecord:
    """表示一个已持久化的 Web 会话。"""

    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MessageRecord:
    """表示一条按顺序存储的 Web 消息信封。"""

    session_id: str
    sequence: int
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RunEvent:
    """表示一条可供 SSE 重放的运行事件。"""

    id: int
    type: str
    data: dict[str, Any]


class SessionRenameRequest(BaseModel):
    """校验会话标题修改请求。"""

    title: str = Field(min_length=1, max_length=100)


class RunCreateRequest(BaseModel):
    """校验新的 Agent 运行请求。"""

    message: str = Field(min_length=1, max_length=100_000)
    mode: str = Field(default="auto", pattern="^(auto|lightweight|full)$")


def derive_session_title(message: str) -> str:
    """根据首条消息生成最多 30 个字符的会话标题。"""
    normalized = message.strip()
    return normalized[:30] or "新对话"


def to_public_message(record: MessageRecord) -> dict[str, Any]:
    """移除消息中仅供模型使用的上下文字段。"""
    payload = dict(record.payload)
    payload.pop("context_messages", None)
    return {
        "sequence": record.sequence,
        "created_at": record.created_at,
        **payload,
    }
