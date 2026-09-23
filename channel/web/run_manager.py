"""MyAgent Web Console 的 Agent 运行协调与事件重放边界。"""

import copy
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from channel.web.schemas import MessageRecord, RunEvent, derive_session_title
from channel.web.store import WebConsoleStore

logger = logging.getLogger(__name__)


class AgentBusyError(RuntimeError):
    """表示共享 Agent 已有正在执行的运行。"""


class RunNotFoundError(LookupError):
    """表示运行编号不存在或已经过期。"""


def rebuild_agent_context(records: list[MessageRecord]) -> list[dict[str, Any]]:
    """拼接完整且兼容模型的历史上下文消息。"""
    context: list[dict[str, Any]] = []
    for record in records:
        if record.payload.get("include_in_context") is True:
            context.extend(
                copy.deepcopy(record.payload.get("context_messages", []))
            )
    return context


def build_user_envelope(message: str, run_id: str) -> dict[str, Any]:
    """构造可展示且会进入 Agent 上下文的用户消息。"""
    return {
        "kind": "user",
        "content": message,
        "thinking": "",
        "execution_events": [],
        "run_id": run_id,
        "status": "complete",
        "include_in_context": True,
        "context_messages": [{"role": "user", "content": message}],
    }


def build_assistant_envelope(
    *,
    run_id: str,
    status: str,
    content: str,
    thinking: str,
    execution_events: list[dict[str, Any]],
    context_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造可持久化的 Assistant 展示与上下文消息信封。"""
    include = status == "complete" and bool(context_messages)
    return {
        "kind": "assistant",
        "content": content,
        "thinking": thinking,
        "execution_events": copy.deepcopy(execution_events),
        "run_id": run_id,
        "status": status,
        "include_in_context": include,
        "context_messages": copy.deepcopy(context_messages) if include else [],
    }


@dataclass
class RunState:
    """保存单次运行的取消、事件重放和终态信息。"""

    run_id: str
    session_id: str
    user_message: str
    baseline_length: int
    max_events: int = 10_000
    cancel_event: threading.Event = field(default_factory=threading.Event)
    events: deque[RunEvent] = field(default_factory=deque)
    next_event_id: int = 1
    finished: bool = False
    terminal_type: str | None = None
    completed_at: float | None = None
    worker: threading.Thread | None = None
    mode: str = "auto"
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        repr=False,
    )

    def publish(self, event_type: str, data: dict[str, Any]) -> RunEvent:
        """保存事件、限制重放数量并唤醒订阅者。"""
        with self._condition:
            event = RunEvent(self.next_event_id, event_type, dict(data))
            self.next_event_id += 1
            self.events.append(event)
            while len(self.events) > self.max_events:
                self.events.popleft()
            if event_type in {"done", "error", "stopped"}:
                self.finished = True
                self.terminal_type = event_type
                self.completed_at = time.monotonic()
            self._condition.notify_all()
            return event

    def events_after(self, last_event_id: int) -> list[RunEvent]:
        """返回编号大于订阅者游标的缓冲事件。"""
        with self._condition:
            return [event for event in self.events if event.id > last_event_id]

    def wait_for_events(
        self,
        last_event_id: int,
        timeout: float,
    ) -> list[RunEvent]:
        """等待可重放事件到达或运行进入终态。"""
        with self._condition:
            available = [event for event in self.events if event.id > last_event_id]
            if not available and not self.finished:
                self._condition.wait(timeout)
            return [event for event in self.events if event.id > last_event_id]

    def request_stop(self) -> None:
        """设置本次运行可重复调用的取消信号。"""
        self.cancel_event.set()


class WebRunManager:
    """协调共享 Agent 的单次运行、事件重放和生命周期。"""

    def __init__(
        self,
        agent,
        store: WebConsoleStore,
        retention_seconds: float = 600.0,
    ):
        """保存共享 Agent、运行注册表和全局执行锁。"""
        self.agent = agent
        self.store = store
        self.retention_seconds = retention_seconds
        self._runs: dict[str, RunState] = {}
        self._active_run_id: str | None = None
        self._lock = threading.RLock()

    def _cleanup_expired(self) -> None:
        """删除超过事件重放保留期的已完成运行。"""
        now = time.monotonic()
        expired = [
            run_id
            for run_id, state in self._runs.items()
            if state.completed_at is not None
            and now - state.completed_at >= self.retention_seconds
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)

    def get_run(self, run_id: str) -> RunState:
        """返回当前或保留期内的运行状态，并先清理过期记录。"""
        with self._lock:
            self._cleanup_expired()
            state = self._runs.get(run_id)
            if state is None:
                raise RunNotFoundError(run_id)
            return state

    def is_session_active(self, session_id: str) -> bool:
        """返回指定会话是否持有全局正在执行的运行。"""
        with self._lock:
            state = self._runs.get(self._active_run_id or "")
            return (
                state is not None
                and state.session_id == session_id
                and not state.finished
            )

    def stop_run(self, run_id: str) -> RunState:
        """请求取消指定运行并返回其当前状态。"""
        state = self.get_run(run_id)
        state.request_stop()
        return state

    def shutdown(self, timeout: float = 10.0) -> None:
        """取消活跃运行，并在宽限期内等待后台线程结束。"""
        with self._lock:
            state = self._runs.get(self._active_run_id or "")
        if state is not None:
            state.request_stop()
            if state.worker is not None:
                state.worker.join(timeout)

    def start_run(self, session_id: str, message: str, mode: str = "auto") -> RunState:
        """保存用户消息并启动唯一允许的 Agent 后台运行。"""
        normalized = message.strip()
        if not normalized:
            raise ValueError("消息不能为空")
        if mode not in {"auto", "lightweight", "full"}:
            raise ValueError("运行模式必须是 auto、lightweight 或 full")
        with self._lock:
            self._cleanup_expired()
            active = self._runs.get(self._active_run_id or "")
            if active is not None and not active.finished:
                raise AgentBusyError("已有其他生成任务正在运行")
            records = self.store.list_messages(session_id)
            context = rebuild_agent_context(records)
            run_id = str(uuid.uuid4())
            if not records:
                self.store.rename_session(
                    session_id,
                    derive_session_title(normalized),
                )
            self.store.append_message(
                session_id,
                build_user_envelope(normalized, run_id),
            )
            self.agent.messages = copy.deepcopy(context)
            state = RunState(
                run_id,
                session_id,
                normalized,
                len(context),
                mode=mode,
            )
            self._runs[run_id] = state
            self._active_run_id = run_id
            state.worker = threading.Thread(
                target=self._execute_run,
                args=(state,),
                name=f"myagent-web-{run_id[:8]}",
                daemon=True,
            )
            state.worker.start()
            return state

    def _assistant_context_delta(self, state: RunState) -> list[dict[str, Any]]:
        """返回持久化用户输入之后新增的 Agent 上下文消息。"""
        for index in range(len(self.agent.messages) - 1, -1, -1):
            message = self.agent.messages[index]
            if (
                message.get("role") == "user"
                and message.get("content") == state.user_message
            ):
                return copy.deepcopy(self.agent.messages[index + 1:])
        return []

    def _persist_terminal_message(
        self,
        state: RunState,
        *,
        status: str,
        content: str,
        thinking: str,
        execution_events: list[dict[str, Any]],
    ) -> None:
        """保存一条已完成、已停止或出错的 Assistant 消息。"""
        context = (
            self._assistant_context_delta(state)
            if status == "complete"
            else []
        )
        self.store.append_message(
            state.session_id,
            build_assistant_envelope(
                run_id=state.run_id,
                status=status,
                content=content,
                thinking=thinking,
                execution_events=execution_events,
                context_messages=context,
            ),
        )

    def _execute_run(self, state: RunState) -> None:
        """消费 Agent 事件、保存终态消息并释放活跃运行。"""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        execution_events: list[dict[str, Any]] = []
        terminal_type = "error"
        terminal_data = {
            "code": "generation_failed",
            "message": "生成失败，请稍后重试。",
        }
        result_answer = ""
        try:
            run_kwargs = {
                "stream": True,
                "cancel_event": state.cancel_event,
            }
            if state.mode != "auto":
                run_kwargs["execution_mode"] = state.mode
            for event in self.agent.run(state.user_message, **run_kwargs):
                event_type = event.get("type")
                if event_type == "text":
                    content = event.get("content", "")
                    text_parts.append(content)
                    state.publish(
                        "text",
                        {"content": content, "step": event.get("step")},
                    )
                elif event_type == "thinking":
                    content = event.get("content", "")
                    thinking_parts.append(content)
                    state.publish(
                        "thinking",
                        {"content": content, "step": event.get("step")},
                    )
                elif event_type in {"tool_start", "tool_end"}:
                    public_event = {
                        key: copy.deepcopy(value)
                        for key, value in event.items()
                        if key != "type"
                    }
                    execution_events.append(
                        {"type": event_type, **copy.deepcopy(public_event)}
                    )
                    state.publish(event_type, public_event)
                elif event_type == "stopped":
                    terminal_type = "stopped"
                    terminal_data = {"status": "stopped"}
                    break
                elif event_type == "done":
                    result = event.get("result")
                    result_answer = getattr(result, "final_answer", "") or ""
                    if result is not None and result.success:
                        terminal_type = "done"
                        terminal_data = {"status": "complete"}
                    break

            status = {
                "done": "complete",
                "stopped": "stopped",
                "error": "error",
            }[terminal_type]
            content = "".join(text_parts) or result_answer
            if not content and terminal_type == "error":
                content = terminal_data["message"]
            try:
                self._persist_terminal_message(
                    state,
                    status=status,
                    content=content,
                    thinking="".join(thinking_parts),
                    execution_events=execution_events,
                )
            except Exception:
                logger.exception("[Web] Failed to persist run %s", state.run_id)
                terminal_type = "error"
                terminal_data = {
                    "code": "persistence_failed",
                    "message": "对话保存失败。",
                }
            state.publish(terminal_type, terminal_data)
        except Exception:
            logger.exception("[Web] Run %s failed", state.run_id)
            try:
                self._persist_terminal_message(
                    state,
                    status="error",
                    content="生成失败，请稍后重试。",
                    thinking="".join(thinking_parts),
                    execution_events=execution_events,
                )
            except Exception:
                logger.exception(
                    "[Web] Failed to persist failed run %s",
                    state.run_id,
                )
            state.publish("error", terminal_data)
        finally:
            with self._lock:
                if self._active_run_id == state.run_id:
                    self._active_run_id = None
