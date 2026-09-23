"""MyAgent Web Console 的会话与消息 HTTP 路由。"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request, Response

from channel.web.auth import require_request_session
from channel.web.schemas import SessionRenameRequest, to_public_message


router = APIRouter(prefix="/api/sessions")


@router.get("")
def list_sessions(request: Request):
    """返回当前浏览器可访问的全部 Web 会话。"""
    require_request_session(request, mutation=False)
    sessions = request.app.state.web_store.list_sessions()
    return {"sessions": [asdict(item) for item in sessions]}


@router.post("")
def create_session(request: Request):
    """创建一个空的 Web 会话。"""
    require_request_session(request, mutation=True)
    return asdict(request.app.state.web_store.create_session())


@router.patch("/{session_id}")
def rename_session(
    session_id: str,
    payload: SessionRenameRequest,
    request: Request,
):
    """修改一个 Web 会话的标题。"""
    require_request_session(request, mutation=True)
    record = request.app.state.web_store.rename_session(session_id, payload.title)
    return asdict(record)


@router.delete("/{session_id}")
def delete_session(session_id: str, request: Request):
    """删除一个未在运行中的 Web 会话。"""
    require_request_session(request, mutation=True)
    if request.app.state.web_runs.is_session_active(session_id):
        raise HTTPException(
            status_code=409,
            detail={"code": "session_running", "message": "该会话正在生成。"},
        )
    request.app.state.web_store.delete_session(session_id)
    return Response(status_code=204)


@router.get("/{session_id}/messages")
def get_messages(session_id: str, request: Request):
    """返回一个会话中可安全展示的历史消息。"""
    require_request_session(request, mutation=False)
    records = request.app.state.web_store.list_messages(session_id)
    return {"messages": [to_public_message(record) for record in records]}
