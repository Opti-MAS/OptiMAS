"""
Path utilities for workspace management.

Provides functions for:
- Session workspace management
- Client path resolution and security validation
- Server path to client path conversion
"""
from pathlib import Path
from typing import Any, Optional

from fastmcp.server.dependencies import get_context, get_http_headers
from mcp import McpError
from mcp.types import ErrorData

from src.middleware.mcp_request_context import get_mcp_tool_session_id_any
from src.tools.common_utils.async_state import get_state_safe
from src.utils.error_codes import INVALID_REQUEST, INVALID_PARAMS


# Global reference to session_middleware (set during tool registration)
_session_middleware = None


def set_session_middleware(session_middleware):
    """Set session middleware reference for utility functions."""
    global _session_middleware
    _session_middleware = session_middleware


def _meta_session_id(meta: Any) -> Optional[str]:
    """Read session_id from MCP meta (object attribute or plain dict)."""
    if meta is None:
        return None
    if isinstance(meta, dict):
        sid = meta.get("session_id")
    else:
        sid = getattr(meta, "session_id", None)
    if sid is None or sid == "":
        return None
    return str(sid).strip()


async def _resolve_session_id_from_mcp_context(ctx: Any) -> Optional[str]:
    """Resolve session_id the same way as SessionMiddleware (state, meta, headers).

    Session middleware sets ``session_id`` on ``MiddlewareContext.fastmcp_context``,
    but tools that call ``get_context()`` sometimes see a different context object
    where that state was never copied. The client still sends ``session_id`` in
    ``call_tool(..., meta=...)``, which is available on ``request_context.meta``.
    """
    mirrored = get_mcp_tool_session_id_any()
    if mirrored:
        return mirrored
    if ctx is None:
        return None
    sid = await get_state_safe(ctx, "session_id")
    if sid:
        return str(sid).strip()

    try:
        rc = getattr(ctx, "request_context", None)
        meta = getattr(rc, "meta", None) if rc is not None else None
        sid = _meta_session_id(meta)
        if sid:
            return sid
    except Exception:
        pass

    try:
        headers = get_http_headers() or {}
        for key, val in headers.items():
            if str(key).lower() == "x-session-id" and val:
                return str(val).strip()
    except Exception:
        pass

    return None


def _hydrate_context_session_id(ctx: Any, session_id: str) -> None:
    """Copy resolved session_id onto context state when middleware state was missing."""
    set_state_fn = getattr(ctx, "set_state", None)
    if not callable(set_state_fn):
        return
    try:
        set_state_fn("session_id", session_id)
    except Exception:
        pass


async def get_session_workspace() -> Path:
    """
    Get session workspace path from context.

    Uses get_context() and ``get_state_safe`` so async FastMCP ``get_state`` is awaited.

    Raises:
        McpError: If session_id is missing or session not found
        RuntimeError: If called outside of a request context
    """
    if _session_middleware is None:
        raise RuntimeError("Session middleware not initialized. Call set_session_middleware() first.")

    try:
        ctx = get_context()
    except Exception as e:
        raise RuntimeError(
            "get_context() failed. This function must be called within a FastMCP request context."
        ) from e

    if ctx is None:
        raise McpError(
            ErrorData(
                code=INVALID_REQUEST,
                message="Session validation failed: Missing context. Please call handshake tool first.",
            )
        )

    session_id = await _resolve_session_id_from_mcp_context(ctx)

    if session_id is None or session_id == "":
        raise McpError(
            ErrorData(
                code=INVALID_REQUEST,
                message="Session validation failed: Missing session ID. Please call handshake tool first.",
            )
        )

    sid = str(session_id).strip()
    _hydrate_context_session_id(ctx, sid)

    session = _session_middleware._sessions.get(sid)

    if not session:
        raise McpError(
            ErrorData(
                code=INVALID_REQUEST,
                message=f"Session validation failed: Session not found or expired: {sid}",
            )
        )

    if not session.workspace_path:
        raise McpError(
            ErrorData(
                code=INVALID_REQUEST,
                message=f"Session validation failed: Workspace not initialized for session: {sid}",
            )
        )

    return session.workspace_path


def _normalize_and_validate_path(client_path: str, workspace: Path) -> Path:
    """
    Normalize and validate a client path relative to workspace.
    
    Internal function that handles:
    - Removing leading slashes (treating /file.txt as file.txt)
    - Detecting path traversal attempts using Path.parts
    - Resolving and validating path is within workspace
    
    Args:
        client_path: Path provided by client
        workspace: Workspace root directory
        
    Returns:
        Resolved Path object within workspace
        
    Raises:
        ValueError: If path is invalid or outside workspace
    """
    # Normalize: treat paths starting with / as relative
    normalized_str = client_path.lstrip('/')
    
    # Convert to Path for component-based validation
    normalized_path = Path(normalized_str)
    
    # Check for path traversal using Path.parts (more reliable than string search)
    if '..' in normalized_path.parts:
        raise ValueError("Path traversal not allowed: '..' is not permitted in paths")
    
    # Check for absolute paths (after normalization)
    if normalized_path.is_absolute():
        raise ValueError("Absolute paths are not allowed. Use relative paths or paths starting with '/' (which will be treated as relative to workspace root).")
    
    # Join with workspace and resolve
    full_path = (workspace / normalized_path).resolve()
    workspace_resolved = workspace.resolve()
    
    # Validate path is within workspace
    try:
        full_path.relative_to(workspace_resolved)
    except ValueError:
        raise ValueError(
            f"Path '{client_path}' resolves to '{full_path}' which is outside workspace '{workspace_resolved}'"
        )
    
    return full_path


async def resolve_workspace_path(client_path: str) -> Path:
    """
    Resolve a client path relative to the current session workspace.
    
    This is the recommended function for resolving client paths in tools.
    It automatically gets the current session workspace and resolves the path.
    
    Client can use absolute paths (e.g., /file.txt) or relative paths (e.g., file.txt).
    Server maps all paths to session workspace.
    
    Examples:
        - Client: /file.txt → Server: {workspace}/file.txt
        - Client: file.txt → Server: {workspace}/file.txt
        - Client: subdir/file.txt → Server: {workspace}/subdir/file.txt
    
    Args:
        client_path: Path provided by client (relative to workspace root)
        
    Returns:
        Resolved Path object within workspace
        
    Raises:
        McpError: If path is invalid or outside workspace, or session issues
    """
    workspace = await get_session_workspace()
    try:
        return _normalize_and_validate_path(client_path, workspace)
    except ValueError as e:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=str(e)
            )
        ) from e


async def get_client_path(server_path: Path) -> str:
    """
    Convert server absolute path to client-visible relative path.
    
    This is the reverse operation of resolve_workspace_path.
    Converts server-side absolute paths back to client-visible paths.
    
    Examples:
        - Server: {workspace}/file.txt → Client: "file.txt"
        - Server: {workspace}/subdir/file.txt → Client: "subdir/file.txt"
    
    Args:
        server_path: Absolute server path (must be within workspace)
        
    Returns:
        Client-visible path relative to workspace root (e.g., "subdir/file.txt")
        
    Raises:
        McpError: If path is outside workspace
    """
    workspace = await get_session_workspace()
    workspace_resolved = workspace.resolve()
    server_path_resolved = Path(server_path).resolve()
    
    try:
        relative = server_path_resolved.relative_to(workspace_resolved)
        return relative.as_posix()  # Use / as separator for cross-platform compatibility
    except ValueError:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"Path '{server_path}' is outside workspace '{workspace}'"
            )
        )

