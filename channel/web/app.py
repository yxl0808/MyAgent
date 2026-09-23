"""MyAgent Web Console 的 FastAPI 应用工厂。"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import create_agent
from channel.web.auth import (
    BrowserSessionManager,
    CsrfError,
    get_request_session,
    set_session_cookie,
)
from channel.web.routes_auth import router as auth_router
from channel.web.routes_runs import router as run_router
from channel.web.routes_sessions import router as session_router
from channel.web.run_manager import AgentBusyError, RunNotFoundError, WebRunManager
from channel.web.store import SessionNotFoundError, WebConsoleStore
from config import conf


STATIC_DIR = Path(__file__).with_name("static")


def create_error_response(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    """构造结构稳定的 Web API 错误响应。"""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """注册已预期 Web 异常的统一 JSON 响应映射。"""

    @app.exception_handler(CsrfError)
    async def handle_csrf(request: Request, exc: CsrfError):
        """将会话、同源或 CSRF 拒绝映射为 403。"""
        return create_error_response(403, "csrf_rejected", "请求来源验证失败。")

    @app.exception_handler(SessionNotFoundError)
    async def handle_missing_session(request: Request, exc: SessionNotFoundError):
        """将不存在会话映射为 404。"""
        return create_error_response(404, "session_not_found", "会话不存在。")

    @app.exception_handler(RunNotFoundError)
    async def handle_missing_run(request: Request, exc: RunNotFoundError):
        """将不存在或过期运行映射为 404。"""
        return create_error_response(404, "run_not_found", "运行不存在或已过期。")

    @app.exception_handler(AgentBusyError)
    async def handle_agent_busy(request: Request, exc: AgentBusyError):
        """将共享 Agent 忙碌映射为 409。"""
        return create_error_response(409, "agent_busy", "已有任务正在生成。")

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        """将 Pydantic 请求校验失败映射为 422。"""
        return create_error_response(422, "invalid_request", "请求参数无效。")

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request, exc: ValueError):
        """将业务参数错误映射为 400。"""
        return create_error_response(400, "invalid_request", "请求参数无效。")

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        """将路由显式异常映射为统一错误结构。"""
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        return create_error_response(
            exc.status_code,
            detail.get("code", "http_error"),
            detail.get("message", "请求失败。"),
        )


def create_web_app(
    config_values: Mapping[str, Any] | None = None,
    *,
    agent=None,
    db_path: str | Path | None = None,
) -> FastAPI:
    """创建支持依赖注入的本机 Web Console 应用。"""
    values = dict(config_values or conf())
    workspace = Path(values.get("agent_workspace", "~/myagent")).expanduser()
    store = WebConsoleStore(db_path or workspace / "web" / "web-console.db")
    web_agent = agent or create_agent()
    browser_sessions = BrowserSessionManager()
    runs = WebRunManager(web_agent, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """初始化持久化边界，并在服务停止时收尾 Agent 线程。"""
        store.initialize()
        yield
        runs.shutdown(timeout=10.0)

    app = FastAPI(title="MyAgent Web Console", lifespan=lifespan)
    register_exception_handlers(app)
    app.state.web_store = store
    app.state.web_auth = browser_sessions
    app.state.web_runs = runs
    app.state.web_show_thinking = bool(values.get("web_show_thinking", False))
    app.include_router(auth_router)
    app.include_router(session_router)
    app.include_router(run_router)
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR, check_dir=False),
        name="static",
    )

    @app.get("/", response_class=FileResponse)
    def serve_root(request: Request):
        """提供本机聊天页面，并在首次访问时建立浏览器会话。"""
        session = get_request_session(request)
        response = FileResponse(STATIC_DIR / "index.html")
        if session is None:
            session = browser_sessions.create_session()
            set_session_cookie(response, session)
        return response

    return app
