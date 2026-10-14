"""
OptiMAS Server entry point.

This module keeps startup orchestration lightweight:
- load config
- initialize logging
- build the FastMCP/FastAPI app
- launch uvicorn
"""
import argparse
import sys, os
import base64
import hmac
import ipaddress
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse
from fastmcp import FastMCP

from src.config import load_settings
from src.middleware import AuthMiddleware, DashboardMiddleware, LoggingMiddleware, SessionMiddleware
from src.resources import register_all_resources
from src.tools import register_all_tools
from src.utils.logger import get_logger, setup_logger

logger = get_logger(__name__)

APP_NAME = "OptiMAS Server"
APP_VERSION = "0.1.0"
MCP_PATH = "/mcp"


def _setup_session_middleware(settings) -> SessionMiddleware:
    return SessionMiddleware(
        timeout_seconds=settings.session.timeout_seconds,
        cleanup_interval=settings.session.cleanup_interval,
        max_sessions=settings.session.max_sessions,
        secret_key=settings.session.secret_key,
    )


def _register_server_components(mcp: FastMCP, settings, session_middleware: SessionMiddleware) -> DashboardMiddleware:
    auth_middleware = AuthMiddleware(api_keys=settings.auth.api_keys)
    mcp.add_middleware(auth_middleware)

    mcp.add_middleware(session_middleware)

    logging_middleware = LoggingMiddleware(
        log_file=settings.logging.tool_calls_log_file,
        enable_file_logging=settings.logging.enable_file_logging,
        enable_context_logging=True,
        session_middleware=session_middleware,
    )
    mcp.add_middleware(logging_middleware)

    server_url = f"http://{settings.server.host}:{settings.server.port}{MCP_PATH}"
    api_key = settings.auth.api_keys[0] if settings.auth.api_keys else ""
    dashboard_middleware = DashboardMiddleware(
        session_middleware=session_middleware,
        server_url=server_url,
        api_key=api_key,
        tool_calls_log_file=settings.logging.tool_calls_log_file,
        fallback_log_file=settings.logging.log_file,
        workspace_root=settings.workspace.path,
    )
    mcp.add_middleware(dashboard_middleware)

    register_all_tools(mcp, session_middleware)
    logger.info("Tools registered")
    register_all_resources(mcp)
    logger.info("Resources registered")
    return dashboard_middleware


_LOOPBACK_HOSTS = {"localhost"}


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _dashboard_allowed(settings) -> bool:
    """Whether dashboard routes may be registered."""
    if not settings.dashboard.enabled:
        logger.info("Dashboard disabled by configuration")
        return False
    if _is_loopback(settings.server.host) or settings.dashboard.allow_remote:
        if not _is_loopback(settings.server.host):
            logger.warning(
                "Dashboard is exposed on a non-loopback bind (%s) via "
                "dashboard.allow_remote. It requires API-key authentication and can execute any "
                "registered tool. Ensure an authenticating proxy sits in front.",
                settings.server.host,
            )
        return True
    logger.warning(
        "Dashboard NOT registered: server is bound to %s (non-loopback) and the "
        "dashboard.allow_remote is false. Set it to true to enable remote access.",
        settings.server.host,
    )
    return False


def _add_service_routes(app: FastAPI, settings, session_middleware: SessionMiddleware) -> None:
    @app.get("/health")
    async def health_check():
        session_count = await session_middleware.get_session_count()
        return {
            "status": "healthy",
            "server": APP_NAME,
            "version": APP_VERSION,
            "mcp_endpoint": MCP_PATH,
            "session_stats": {
                "total_sessions": session_count,
                "max_sessions": settings.session.max_sessions if settings.session.max_sessions > 0 else "unlimited",
            },
        }

    if not settings.server.debug:
        return

    @app.get("/debug/routes")
    async def debug_routes():
        routes = []
        for route in app.routes:
            if hasattr(route, "path"):
                route_info = {"path": route.path, "name": getattr(route, "name", None)}
                if hasattr(route, "methods"):
                    route_info["methods"] = list(route.methods) if route.methods else None
                routes.append(route_info)
        return {"routes": routes, "total": len(routes)}


def create_app(config_path: str | Path = "config.yaml", settings=None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or load_settings(config_path)

    setup_logger(
        name="optimas_server",
        log_file=settings.logging.log_file,
        log_level=settings.logging.log_level,
        enable_file_logging=settings.logging.enable_file_logging,
        enable_console_logging=settings.logging.enable_console_logging,
    )

    logger.info("Starting %s...", APP_NAME)
    logger.info("Server: %s:%s", settings.server.host, settings.server.port)

    if not _is_loopback(settings.server.host) and settings.session.is_placeholder_secret():
        raise RuntimeError(
            "session.secret_key is still the shipped placeholder while the server is "
            "bound to a non-loopback address. Session IDs would be forgeable. Generate "
            "one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )

    if not settings.auth.api_keys:
        raise RuntimeError("At least one MCP API key is required.")
    if not _is_loopback(settings.server.host) and "optimas-local-dev-key" in settings.auth.api_keys:
        raise RuntimeError("Replace the development API key before binding to a non-loopback address.")

    mcp = FastMCP(APP_NAME)
    session_middleware = _setup_session_middleware(settings)
    dashboard_middleware = _register_server_components(mcp, settings, session_middleware)

    mcp_app = mcp.http_app(path=MCP_PATH)

    @asynccontextmanager
    async def lifespan(app):
        async with mcp_app.lifespan(app):
            try:
                yield
            finally:
                await session_middleware.shutdown()

    app = FastAPI(
        title=APP_NAME,
        description="MCP Server",
        version=APP_VERSION,
        routes=[*mcp_app.routes],
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def authenticate_http(request, call_next):
        path = request.url.path
        is_mcp = path == MCP_PATH or path.startswith(MCP_PATH + "/")
        is_dashboard = path == "/dashboard" or path.startswith("/api/")
        if is_mcp or is_dashboard:
            key = request.headers.get("X-MCP-Key", "")
            if is_dashboard and not key:
                authorization = request.headers.get("Authorization", "")
                if authorization.lower().startswith("basic "):
                    try:
                        credentials = base64.b64decode(authorization.split(" ", 1)[1], validate=True).decode()
                        key = credentials.split(":", 1)[1]
                    except (ValueError, UnicodeError, IndexError):
                        key = ""
            if not any(hmac.compare_digest(key.encode(), valid.encode()) for valid in settings.auth.api_keys):
                headers = {"WWW-Authenticate": 'Basic realm="OptiMAS"'} if is_dashboard else {}
                return JSONResponse({"detail": "Valid MCP API key required."}, status_code=401, headers=headers)
        return await call_next(request)

    if _dashboard_allowed(settings):
        dashboard_middleware.register_routes(app)
    _add_service_routes(app, settings, session_middleware)

    logger.info("FastAPI application created")
    logger.info("MCP endpoint: %s", MCP_PATH)
    if _dashboard_allowed(settings):
        logger.info("Dashboard: http://%s:%s/dashboard", settings.server.host, settings.server.port)
    logger.info("API docs: /docs")
    logger.info("Health check: /health")
    if settings.server.debug:
        logger.info("Debug routes: /debug/routes")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OptiMAS Server - MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config)
        settings = load_settings(config_path)
        app = create_app(config_path, settings=settings)
        logger.info("Starting server on %s:%s", settings.server.host, settings.server.port)
        uvicorn.run(
            app,
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.logging.log_level.lower() if settings.server.debug else "info",
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as exc:
        logger.error("Server error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

