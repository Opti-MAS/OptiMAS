"""
Session management tools.

Provides handshake tool for establishing sessions.
"""
# Note: FastMCP instance and dependencies will be passed when registering tools
_mcp_instance = None
_session_middleware = None


def register_tools(mcp, session_middleware):
    """
    Register session tools to FastMCP instance.
    
    Args:
        mcp: FastMCP instance
        session_middleware: SessionMiddleware instance
    """
    global _mcp_instance, _session_middleware
    _mcp_instance = mcp
    _session_middleware = session_middleware

    @mcp.tool
    async def handshake(task_id: str, ctx=None, workspace_subdir: str | None = None, skip_session_dir: bool = False, docker_container_id: str | None = None, log_base_path: str | None = None, logical_task_id: str | None = None) -> dict:
        """
        Establish a session by providing task_id.
        
        This tool generates a session_id based on task_id and server secret key.
        The session_id format is: {task_id}_{32-char-signature}.
        
        Supports overriding workspace *root* (server-side managed) to place session directories
        under a specific subfolder of the server workspace root.
        
        - Default: workspace_root = settings.workspace.path, creates {workspace_root}/{session_id}
        - If workspace_subdir provided: workspace_root = settings.workspace.path / workspace_subdir,
          creates {workspace_root}/{session_id}
        - If skip_session_dir=True, does NOT create the {session_id} subfolder, using workspace_root directly.
        
        Security model:
        - The server still creates the final per-session directory ({workspace_root}/{session_id})
        - Tools remain confined to session.workspace_path via resolve_workspace_path()
        - workspace_subdir is validated to be a *relative* safe path (no absolute paths / traversal)
        
        Args:
            task_id: Task identifier provided by client
            workspace_subdir: Optional subdirectory under server workspace root to place this session.
                              Example: "runs/{run_id}/mas" (the session dir becomes .../runs/{run_id}/mas/{session_id})
            skip_session_dir: If True, skips appending {session_id} to the workspace path.
            docker_container_id: Optional Docker container ID.
            ctx: FastMCP Context (injected automatically)
            
        Returns:
            Dictionary with session_id and workspace_path:
            {
                "session_id": "task123_abcd",
                "task_id": "task123",
                "workspace_path": "runs/run_id/mas/state_id/problem_id/session_id"  # Relative path (relative to server workspace base root)
            }
            
        Raises:
            McpError: If session creation fails
        """
        from mcp import McpError
        from mcp.types import ErrorData
        from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR
        from pathlib import Path
        
        if not task_id or not task_id.strip():
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message="Invalid parameters: task_id is required and cannot be empty",
                )
            )
        
        task_id = task_id.strip()
        
        try:
            # Generate session_id from task_id
            session_id = _session_middleware.generate_session_id(task_id, _session_middleware.secret_key)

            from src.config import get_settings
            settings = get_settings()
            base_root = Path(settings.workspace.path).resolve()

            if workspace_subdir is not None:
                sub = workspace_subdir.strip()
                if not sub:
                    raise ValueError("workspace_subdir cannot be empty if provided")
                sub_path = Path(sub)
                if sub_path.is_absolute() or ".." in sub_path.parts:
                    raise ValueError("workspace_subdir must be a safe relative path (no absolute paths or '..').")
                workspace_root = (base_root / sub_path).resolve()
                workspace_root.relative_to(base_root)
                metadata = {"created_via": "handshake", "workspace_mode": "server-side", "workspace_subdir": sub, "skip_session_dir": skip_session_dir}
            else:
                workspace_root = base_root
                metadata = {"created_via": "handshake", "workspace_mode": "server-side", "skip_session_dir": skip_session_dir}

            if logical_task_id:
                metadata["logical_task_id"] = logical_task_id.strip()
            if docker_container_id:
                metadata["docker_container_id"] = docker_container_id.strip()
            if log_base_path:
                lp = Path(log_base_path.strip())
                if lp.is_absolute() or ".." in lp.parts:
                    raise ValueError("log_base_path must be a safe relative path.")
                lp = (workspace_root / lp).resolve()
                lp.relative_to(workspace_root)
                metadata["log_base_path"] = str(lp)

            session = await _session_middleware.create_session(
                task_id=task_id,
                session_id=session_id,
                metadata=metadata,
                workspace_root=str(workspace_root),
                skip_session_dir=skip_session_dir,
            )
            
            # Calculate workspace relative path (relative to server workspace base root)
            workspace_relative_path = None
            if session.workspace_path:
                try:
                    # workspace_path is absolute; compute the path relative to base_root
                    workspace_relative_path = str(session.workspace_path.relative_to(base_root))
                except ValueError:
                    # If workspace_path is not under base_root, fall back to absolute path
                    workspace_relative_path = str(session.workspace_path)
            
            return {
                "success": True,
                "data": {
                    "session_id": session.id,
                    "task_id": session.task_id,
                    "workspace_path": workspace_relative_path,  # Relative path (relative to server workspace base root)
                },
                "error": None,
            }
        except RuntimeError as e:
            # Re-raise session limit errors so client gets clear message
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=str(e),
                )
            )
        except ValueError as e:
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Invalid parameters: Failed to create session - {str(e)}",
                )
            )
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Internal error: Failed to create session - {str(e)}",
                )
            )

    @mcp.tool
    async def close_session() -> dict:
        """
        Close the current session and free server resources.

        The session to close is taken from the current request: either from context state
        (set by middleware from meta.session_id) or from X-Session-ID header.
        Call this when the client is done with the task so the server can reclaim memory.
        After closing, the session is removed; further requests with this session_id will fail.
        """
        from fastmcp.server.dependencies import get_context, get_http_headers
        from src.tools.common_utils.async_state import get_state_safe
        from mcp import McpError
        from mcp.types import ErrorData
        from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR

        session_id = None
        try:
            # Prefer context state (set by middleware from meta.session_id); fallback to headers
            ctx = get_context()
            if ctx is not None:
                try:
                    session_id = await get_state_safe(ctx, "session_id")
                except Exception:
                    pass
                if not session_id:
                    try:
                        request_context = getattr(ctx, "request_context", None)
                        meta = getattr(request_context, "meta", None) if request_context is not None else None
                        if meta is not None and hasattr(meta, "session_id"):
                            session_id = getattr(meta, "session_id", None)
                    except Exception:
                        pass
            if not session_id:
                headers = get_http_headers()
                if headers:
                    session_id = headers.get("X-Session-ID") or headers.get("x-session-id")
            if not session_id:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="close_session requires an active session (session_id in meta or X-Session-ID header).",
                    )
                )
            session_id = str(session_id).strip()
            deleted = await _session_middleware.delete_session(session_id)
            if deleted:
                try:
                    from src.tools.workbench_ops.workbench_tools import on_session_closed
                    on_session_closed(session_id)
                except Exception:
                    pass
            return {
                "success": deleted,
                "data": {
                    "session_id": session_id,
                    "message": f"Session {session_id} closed." if deleted else "Session not found or already closed.",
                } if deleted else None,
                "error": None if deleted else "Session not found or already closed.",
            }
        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to close session: {str(e)}",
                )
            )

