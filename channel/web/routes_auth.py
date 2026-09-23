"""MyAgent Web Console 的本机浏览器会话路由。"""

from fastapi import APIRouter, Request, Response

from channel.web.auth import get_request_session, set_session_cookie


router = APIRouter(prefix="/api/auth")


@router.get("/status")
def auth_status(request: Request, response: Response):
    """返回本机浏览器会话状态，缺失时自动创建会话。"""
    session = get_request_session(request)
    if session is None:
        session = request.app.state.web_auth.create_session()
        set_session_cookie(response, session)
    return {
        "local_only": True,
        "csrf_token": session.csrf_token,
        "show_thinking_default": request.app.state.web_show_thinking,
    }
