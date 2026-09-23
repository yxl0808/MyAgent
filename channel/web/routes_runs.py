"""MyAgent Web Console 的运行创建、停止和 SSE 事件路由。"""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from channel.web.auth import require_request_session
from channel.web.run_manager import RunState
from channel.web.schemas import RunCreateRequest


router = APIRouter(prefix="/api")


@router.post("/sessions/{session_id}/runs", status_code=202)
def create_run(session_id: str, payload: RunCreateRequest, request: Request):
    """启动唯一允许的 Agent 生成任务。"""
    require_request_session(request, mutation=True)
    state = request.app.state.web_runs.start_run(
        session_id,
        payload.message,
        mode=payload.mode,
    )
    return {"run_id": state.run_id, "mode": state.mode}


def _encode_sse_event(
    *,
    event: str,
    data: str,
    event_id: str | None = None,
) -> str:
    """将一个命名事件编码为浏览器 EventSource 可读取的 SSE 文本。"""
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {line}" for line in data.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


async def _event_stream(state: RunState, last_event_id: int):
    """重放缓冲事件，并每 15 秒发送命名心跳事件。"""
    cursor = last_event_id
    while True:
        events = await asyncio.to_thread(state.wait_for_events, cursor, 15.0)
        if not events:
            if state.finished:
                return
            yield _encode_sse_event(event="heartbeat", data="{}")
            continue
        for event in events:
            cursor = event.id
            yield _encode_sse_event(
                event=event.type,
                data=json.dumps(event.data, ensure_ascii=False),
                event_id=str(event.id),
            )
            if event.type in {"done", "error", "stopped"}:
                return


@router.get("/runs/{run_id}/events")
def run_events(run_id: str, request: Request):
    """打开一个可重放的本机浏览器 SSE 运行事件流。"""
    require_request_session(request, mutation=False)
    state = request.app.state.web_runs.get_run(run_id)
    try:
        cursor = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        cursor = 0
    return StreamingResponse(
        _event_stream(state, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str, request: Request):
    """对活跃或已完成运行请求可重复执行的协作停止。"""
    require_request_session(request, mutation=True)
    state = request.app.state.web_runs.stop_run(run_id)
    return {
        "run_id": state.run_id,
        "finished": state.finished,
        "status": state.terminal_type,
    }
