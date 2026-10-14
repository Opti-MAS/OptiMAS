"""
Session management middleware.

Validates session for tool calls (except handshake).
Session is created via handshake tool.
"""
import asyncio
import hashlib
import logging
import hmac
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers
from mcp import McpError
from mcp.types import ErrorData

logger = logging.getLogger(__name__)

from src.middleware.mcp_request_context import (
    attach_tool_session_to_current_task,
    clear_tool_session_for_task,
    pop_tool_session_id,
    push_tool_session_id,
)
from src.tools.common_utils.async_state import get_state_safe
from src.utils.error_codes import INVALID_REQUEST, INVALID_PARAMS


async def resolve_tool_call_session_id(context: MiddlewareContext) -> Optional[str]:
    """Resolve MCP session id for the current tool call.

    Order matches :meth:`SessionMiddleware._get_session_id` (state, meta, ``get_context``, headers).

    **Why this exists:** ``LoggingMiddleware`` may run *outside* ``SessionMiddleware``. After the
    inner chain returns, ``SessionMiddleware`` clears task-local session mirrors in ``finally``,
    and FastMCP may clear context ``session_id`` state — but the client-supplied ``meta.session_id``
    on this request's ``MiddlewareContext`` remains available for audit logging.
    """
    if hasattr(context, "fastmcp_context") and context.fastmcp_context is not None:
        try:
            session_id = await get_state_safe(context.fastmcp_context, "session_id")
            if session_id:
                return str(session_id).strip()
        except Exception:
            pass

    if hasattr(context, "fastmcp_context") and context.fastmcp_context is not None:
        try:
            request_context = getattr(context.fastmcp_context, "request_context", None)
            if request_context:
                meta = getattr(request_context, "meta", None)
                if meta is not None:
                    if isinstance(meta, dict):
                        session_id = meta.get("session_id")
                    else:
                        session_id = getattr(meta, "session_id", None)
                    if session_id:
                        return str(session_id).strip()
        except Exception:
            pass

    try:
        from fastmcp.server.dependencies import get_context

        ctx = get_context()
        if ctx is not None:
            try:
                session_id = await get_state_safe(ctx, "session_id")
                if session_id:
                    return str(session_id).strip()
            except Exception:
                pass
            try:
                request_context = getattr(ctx, "request_context", None)
                meta = getattr(request_context, "meta", None) if request_context is not None else None
                if meta is not None:
                    if isinstance(meta, dict):
                        session_id = meta.get("session_id")
                    else:
                        session_id = getattr(meta, "session_id", None)
                    if session_id:
                        return str(session_id).strip()
            except Exception:
                pass
    except Exception:
        pass

    try:
        headers = get_http_headers()
        headers_lower = {k.lower(): v for k, v in headers.items()}
        session_id = headers_lower.get("x-session-id")
        if session_id:
            return str(session_id).strip()
    except Exception:
        pass

    return None


@dataclass
class Session:
    """Session data class"""

    id: str
    task_id: Optional[str]
    created_at: datetime
    last_accessed: datetime
    metadata: Dict = field(default_factory=dict)
    workspace_path: Optional[Path] = None

    def is_expired(self, timeout_seconds: int) -> bool:
        """
        Check if session is expired.

        Args:
            timeout_seconds: Session timeout in seconds

        Returns:
            True if session is expired
        """
        if timeout_seconds <= 0:
            return False  # No expiration
        elapsed = (datetime.now() - self.last_accessed).total_seconds()
        return elapsed > timeout_seconds

    def to_dict(self) -> dict:
        """Convert session to dictionary"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata,
        }


class SessionMiddleware(Middleware):
    """
    Middleware for session management.

    Features:
    - Validates session for tool calls (except handshake)
    - Uses on_call_tool hook to only validate tool execution operations
    - Session is created via handshake tool
    - Updates session last_accessed timestamp
    - Manages session storage and lifecycle internally
    
    Note: Other operations (list, resources/read, etc.) don't require session validation.
    """

    def __init__(
        self,
        timeout_seconds: int = 7200,
        cleanup_interval: int = 300,
        max_sessions: int = 0,
        secret_key: str = "optimas-server-secret-key-change-in-production",
    ):
        """
        Initialize session middleware.

        Args:
            timeout_seconds: Session timeout in seconds (0 = no expiration)
            cleanup_interval: Cleanup interval in seconds
            max_sessions: Maximum number of sessions (0 = unlimited)
            secret_key: Secret key for generating session_id (HMAC key)
        """
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self.timeout_seconds = timeout_seconds
        self.cleanup_interval = cleanup_interval
        self.max_sessions = max_sessions
        self.secret_key = secret_key
        self._cleanup_task: Optional[asyncio.Task] = None

    def _drop_session(self, session_id: str) -> bool:
        """Remove session metadata and its benchmark state while holding _lock."""
        removed = self._sessions.pop(session_id, None)
        if removed is not None:
            from src.tools.workbench_ops.session_store import WorkbenchSessionStore
            WorkbenchSessionStore.remove(session_id)
        return removed is not None

    @staticmethod
    def generate_session_id(task_id: str, secret_key: str) -> str:
        """
        Generate session_id from task_id using HMAC-SHA256.
        
        Format: {task_id}_{32-char-signature}.
        
        Args:
            task_id: Task ID
            secret_key: Secret key for HMAC
            
        Returns:
            Session ID in format: {task_id}_{secret}
        """
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", task_id):
            raise ValueError("task_id must be a safe identifier (1-200 letters, digits, '.', '_' or '-').")
        hmac_obj = hmac.new(
            secret_key.encode('utf-8'),
            task_id.encode('utf-8'),
            hashlib.sha256
        )
        secret = hmac_obj.hexdigest()[:32]
        return f"{task_id}_{secret}"
    
    @staticmethod
    def validate_session_id(session_id: str, secret_key: str) -> Tuple[Optional[str], bool]:
        """
        Validate session_id and extract task_id.
        
        Args:
            session_id: Session ID to validate (format: {task_id}_{secret})
            secret_key: Secret key for HMAC
            
        Returns:
            Tuple of (task_id, is_valid)
            - If valid: (task_id, True)
            - If invalid: (None, False)
        """
        try:
            # Parse session_id: {task_id}_{secret}
            if '_' not in session_id:
                return None, False
            
            parts = session_id.rsplit('_', 1)  # Split from right to handle task_id with underscores
            if len(parts) != 2:
                return None, False
            
            task_id, provided_secret = parts
            
            # Recompute secret
            expected_session_id = SessionMiddleware.generate_session_id(task_id, secret_key)
            expected_parts = expected_session_id.rsplit('_', 1)
            if len(expected_parts) != 2:
                return None, False
            
            expected_secret = expected_parts[1]
            
            # Compare secrets (constant-time comparison to prevent timing attacks)
            is_valid = hmac.compare_digest(provided_secret, expected_secret)
            
            if is_valid:
                return task_id, True
            else:
                return None, False
        except Exception:
            return None, False

    async def create_session(
        self, 
        task_id: Optional[str] = None, 
        metadata: Optional[Dict] = None,
        session_id: Optional[str] = None,
        workspace_root: Optional[str | Path] = None,
        skip_session_dir: bool = False
    ) -> Session:
        """
        Create a new session.

        Args:
            task_id: Optional task ID associated with session
            metadata: Optional metadata dictionary
            session_id: Optional pre-generated session ID (if provided, task_id must match)
            workspace_root: Optional root directory for session workspace. If provided, creates {workspace_root}/{session_id} directory.
            skip_session_dir: If True, skips appending {session_id} to the workspace path.

        Returns:
            Created Session object

        Raises:
            ValueError: If session_id is provided but doesn't match task_id

        Active sessions are preserved when the configured capacity is reached.
        """
        async with self._lock:
            for expired_id in [sid for sid, session in self._sessions.items()
                               if session.is_expired(self.timeout_seconds)]:
                self._drop_session(expired_id)
            if self.max_sessions > 0 and len(self._sessions) >= self.max_sessions:
                raise RuntimeError("Maximum active sessions reached; close a session or wait for expiry.")

            # Generate or validate session ID
            if session_id is None:
                if task_id:
                    # Generate session_id from task_id
                    session_id = self.generate_session_id(task_id, self.secret_key)
                else:
                    # Generate random UUID for backward compatibility
                    session_id = str(uuid.uuid4())
            else:
                # Validate provided session_id matches task_id
                if task_id:
                    expected_task_id, is_valid = self.validate_session_id(session_id, self.secret_key)
                    if not is_valid or expected_task_id != task_id:
                        raise ValueError(
                            f"Session ID {session_id} does not match task_id {task_id}"
                        )

            # Explicit workspace path: use directly as session workspace (e.g. optimizer sees exact step dir)
            workspace_path = None
            if workspace_root is not None:
                workspace_root_path = Path(workspace_root).resolve()
                workspace_root_path.mkdir(parents=True, exist_ok=True)
                use_skip_session_dir = skip_session_dir or (metadata and metadata.get("skip_session_dir", False))
                if use_skip_session_dir:
                    workspace_path = workspace_root_path
                else:
                    # Create session-specific workspace directory
                    workspace_path = (workspace_root_path / session_id).resolve()
                    workspace_path.relative_to(workspace_root_path)
                    workspace_path.mkdir(parents=True, exist_ok=True)

            # Create session
            now = datetime.now()
            session = Session(
                id=session_id,
                task_id=task_id,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
                workspace_path=workspace_path,
            )

            # Store session
            self._sessions[session_id] = session

            # Start cleanup task if not already running
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session object if found and not expired, None otherwise
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            # Check if expired
            if session.is_expired(self.timeout_seconds):
                self._drop_session(session_id)
                return None

            return session

    async def update_session(self, session_id: str) -> bool:
        """
        Update session's last_accessed timestamp.

        Args:
            session_id: Session ID

        Returns:
            True if session was updated, False if not found or expired
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False

            # Check if expired
            if session.is_expired(self.timeout_seconds):
                self._drop_session(session_id)
                return False

            # Update last accessed time
            session.last_accessed = datetime.now()
            return True

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if session was deleted, False if not found
        """
        async with self._lock:
            if session_id in self._sessions:
                self._drop_session(session_id)
                return True
            return False

    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        async with self._lock:
            expired_ids = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_expired(self.timeout_seconds)
            ]

            for session_id in expired_ids:
                self._drop_session(session_id)

            return len(expired_ids)

    async def list_sessions(self) -> List[Session]:
        """
        List all active sessions.

        Returns:
            List of Session objects
        """
        async with self._lock:
            # Filter out expired sessions
            active_sessions = [
                session
                for session in self._sessions.values()
                if not session.is_expired(self.timeout_seconds)
            ]
            return active_sessions.copy()

    async def get_session_count(self) -> int:
        """
        Get number of active sessions.

        Returns:
            Number of active (non-expired) sessions
        """
        async with self._lock:
            return sum(
                1
                for session in self._sessions.values()
                if not session.is_expired(self.timeout_seconds)
            )

    async def _cleanup_loop(self):
        """Background task to periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                # Never let a cleanup failure kill the loop.
                logger.exception("Session cleanup pass failed; continuing")

    async def shutdown(self):
        """Shutdown session manager and cleanup background tasks"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def on_call_tool(
        self, context: MiddlewareContext, call_next
    ) -> any:
        """Validate session for tool calls (except handshake)."""
        tool_name = context.message.name
        
        # Skip handshake tool (it creates the session)
        if tool_name == 'handshake':
            return await call_next(context)

        # Get session ID from meta or headers
        session_id = await self._get_session_id(context)
        if not session_id:
            raise McpError(
                ErrorData(
                    code=INVALID_REQUEST,
                    message="Session validation failed: Missing session ID. Please call handshake tool first and include session_id in meta parameter.",
                )
            )

        # Validate session_id format and extract task_id
        task_id, is_valid = self.validate_session_id(
            session_id, self.secret_key
        )
        if not is_valid:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Session validation failed: Invalid session ID format: {session_id}"
                )
            )
        
        # Get and verify session
        session = await self.get_session(session_id)
        if not session:
            raise McpError(
                ErrorData(
                    code=INVALID_REQUEST,
                    message=f"Session validation failed: Session not found or expired: {session_id}"
                )
            )
        
        if session.task_id != task_id:
            raise McpError(
                ErrorData(
                    code=INVALID_REQUEST,
                    message=f"Session validation failed: Session task_id mismatch. Expected {task_id}, got {session.task_id}",
                )
            )

        # Update last accessed time
        await self.update_session(session_id)

        # Store in context state
        if hasattr(context, 'fastmcp_context') and context.fastmcp_context is not None:
            try:
                context.fastmcp_context.set_state("session_id", session_id)
                context.fastmcp_context.set_state("task_id", task_id)
            except Exception:
                pass

        # Tools that use get_context() may not see the same fastmcp_context; mirror
        # session id for path_utils (Task-keyed map + ContextVar — see mcp_request_context).
        task_key = attach_tool_session_to_current_task(session_id)
        cv_token = push_tool_session_id(session_id)
        try:
            return await call_next(context)
        finally:
            clear_tool_session_for_task(task_key)
            pop_tool_session_id(cv_token)

    async def _get_session_id(self, context: MiddlewareContext) -> str | None:
        """Get session ID from context state, meta, or HTTP headers."""
        return await resolve_tool_call_session_id(context)
