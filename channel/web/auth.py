"""MyAgent Web Console 的本地浏览器请求保护边界。"""

import secrets
import threading
from dataclasses import dataclass
from urllib.parse import urlsplit

from fastapi import Request, Response


SESSION_COOKIE = "myagent_web_session"
CSRF_HEADER = "x-csrf-token"


class CsrfError(PermissionError):
    """表示浏览器请求缺少或携带了错误的 CSRF 凭证。"""


@dataclass(frozen=True)
class BrowserSession:
    """保存一个本地浏览器会话及其 CSRF 凭证。"""

    token: str
    csrf_token: str


def set_session_cookie(response: Response, session: BrowserSession) -> None:
    """为本机浏览器会话写入 host-only 的 HttpOnly Cookie。"""
    response.set_cookie(
        SESSION_COOKIE,
        session.token,
        httponly=True,
        samesite="strict",
        secure=False,
        path="/",
    )


def get_request_session(request: Request) -> BrowserSession | None:
    """从当前请求 Cookie 中恢复本地浏览器会话。"""
    return request.app.state.web_auth.get_session(
        request.cookies.get(SESSION_COOKIE)
    )


def require_request_session(
    request: Request,
    *,
    mutation: bool,
) -> BrowserSession:
    """为本机浏览器请求执行会话、同源和 CSRF 检查。"""
    session = get_request_session(request)
    if session is None:
        raise CsrfError("浏览器会话不存在或已过期")
    if mutation:
        require_same_origin(
            request.headers.get("origin"),
            request.headers.get("host"),
        )
        request.app.state.web_auth.require_csrf(
            session,
            request.headers.get(CSRF_HEADER),
        )
    return session


class BrowserSessionManager:
    """管理当前进程中的本地浏览器会话。"""

    def __init__(self):
        """创建空会话集合和可重入线程锁。"""
        self._sessions: dict[str, BrowserSession] = {}
        self._lock = threading.RLock()

    def create_session(self) -> BrowserSession:
        """创建并保存一个具有随机凭证的本地浏览器会话。"""
        session = BrowserSession(
            token=secrets.token_urlsafe(32),
            csrf_token=secrets.token_urlsafe(32),
        )
        with self._lock:
            self._sessions[session.token] = session
        return session

    def get_session(self, token: str | None) -> BrowserSession | None:
        """根据有效令牌返回本地浏览器会话。"""
        if not token:
            return None
        with self._lock:
            return self._sessions.get(token)

    def require_csrf(
        self,
        session: BrowserSession,
        supplied: str | None,
    ) -> None:
        """使用恒定时间比较校验浏览器提交的 CSRF 令牌。"""
        if not supplied or not secrets.compare_digest(
            supplied.encode("utf-8"),
            session.csrf_token.encode("utf-8"),
        ):
            raise CsrfError("无效的 CSRF 令牌")


def require_same_origin(origin: str | None, host: str | None) -> None:
    """拒绝并非来自当前 Web Console 地址的修改请求。"""
    if not origin or not host:
        raise CsrfError("请求缺少来源信息")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != host:
        raise CsrfError("拒绝跨来源请求")
