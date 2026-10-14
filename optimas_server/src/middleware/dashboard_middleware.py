"""
Dashboard middleware for OptiMAS Server.

Provides Web UI and API endpoints for monitoring and managing the server.
This middleware handles Dashboard-related HTTP requests and provides
methods to register Dashboard routes with FastAPI.
"""
import asyncio
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from src.middleware.session_middleware import SessionMiddleware
from src.utils.logger import get_logger

logger = get_logger(__name__)

# When scanning workspace for tool_call/tool_calls.jsonl, skip subtrees that are
# large or never contain MCP tool logs (full rglob on workspace is too slow).
_TOOL_TRACE_PRUNE_SUBDIRS = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        "target",
        "model",
        ".cache",
        "public",
        "resources",
        "working_trajectory",
        "workspace",
        "draft",
    }
)


class DashboardMiddleware(Middleware):
    """
    Middleware for Dashboard functionality.
    
    This middleware:
    1. Provides Dashboard API endpoints via FastAPI routes
    2. Serves Dashboard HTML interface
    3. Manages Dashboard session and tool execution
    
    Note: Dashboard routes are registered with FastAPI, not handled via
    MCP middleware hooks. This middleware primarily provides initialization
    and route registration functionality.
    """

    def __init__(
        self,
        session_middleware: SessionMiddleware,
        server_url: str,
        api_key: str,
        tool_calls_log_file: str = "logs/mcp_server_tools.jsonl",
        fallback_log_file: Optional[str] = None,
        workspace_root: str = "./workspace",
        dashboard_path: Optional[Path] = None,
    ):
        """
        Initialize Dashboard middleware.

        Args:
            session_middleware: SessionMiddleware instance
            server_url: MCP server URL (e.g., "http://127.0.0.1:8188/mcp")
            api_key: API key for authentication
            tool_calls_log_file: JSONL file written by LoggingMiddleware (primary source for history)
            fallback_log_file: Optional legacy log path (e.g. mcp_server.log) to parse JSON lines for old runs
            workspace_root: Server workspace root (scan for **/tool_call/tool_calls.jsonl)
            dashboard_path: Path to dashboard HTML file
        """
        self.session_middleware = session_middleware
        self.server_url = server_url
        self.api_key = api_key
        self.tool_calls_log_file = tool_calls_log_file
        self.fallback_log_file = fallback_log_file
        self.workspace_root = workspace_root
        self.dashboard_path = dashboard_path or (Path(__file__).parent.parent.parent / "dashboard" / "index.html")
        
        # Default task_id for dashboard session (consistent across all dashboard tool executions)
        self.default_task_id = "dashboard"
        
        # Internal MCP client for tool operations
        # We need separate clients for each session since headers are set at transport creation
        self._clients: Dict[str, Client] = {}  # session_id -> Client
        self._transports: Dict[str, StreamableHttpTransport] = {}  # session_id -> Transport
        
        # Cache for default dashboard session_id
        self._default_session_id: Optional[str] = None

    def register_routes(self, app: FastAPI):
        """
        Register Dashboard routes with FastAPI application.
        
        This method should be called in main.py after creating the FastAPI app.
        
        Args:
            app: FastAPI application instance
        """
        # Dashboard HTML routes
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_root():
            """Serve dashboard HTML at root"""
            return self._serve_dashboard_html()
        
        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """Serve dashboard HTML"""
            return self._serve_dashboard_html()

        # Dashboard API routes
        @app.get("/api/tools")
        async def get_tools() -> Dict[str, Any]:
            """Get all available tools"""
            return await self._get_tools()

        @app.get("/api/sessions")
        async def get_sessions() -> Dict[str, Any]:
            """Get all active sessions"""
            return await self._get_sessions()

        @app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str) -> Dict[str, Any]:
            """Delete a session"""
            return await self._delete_session(session_id)

        @app.get("/api/history")
        async def get_history(
            session_id: Optional[str] = Query(None, description="Session ID (from server log)"),
            workspace_log: Optional[str] = Query(
                None,
                description="Relative path under workspace to tool_call/tool_calls.jsonl",
            ),
            limit: int = Query(100, ge=1, le=2000),
        ) -> Dict[str, Any]:
            """Get tool call history from server JSONL or a workspace-local tool_calls.jsonl"""
            return await self._get_history(session_id, workspace_log, limit)

        @app.get("/api/workspace-traces")
        async def get_workspace_traces() -> Dict[str, Any]:
            """List discovered tool_call/tool_calls.jsonl under workspace (for offline / per-query traces)"""
            return await self._get_workspace_traces()

        @app.post("/api/execute-tool")
        async def execute_tool(request: Request) -> Dict[str, Any]:
            """Execute a tool from the dashboard"""
            return await self._execute_tool(request)

        logger.info("Dashboard routes registered")

    def _serve_dashboard_html(self) -> HTMLResponse:
        """Serve dashboard HTML file"""
        try:
            if not self.dashboard_path.exists():
                logger.error(f"Dashboard not found at {self.dashboard_path}")
                return HTMLResponse(
                    content="<h1>Dashboard not found</h1><p>Please ensure dashboard/index.html exists.</p>",
                    status_code=404,
                )

            with open(self.dashboard_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            return HTMLResponse(content=html_content)
        except Exception as e:
            logger.error(f"Failed to serve dashboard: {e}", exc_info=True)
            return HTMLResponse(
                content=f"<h1>Error</h1><p>Failed to load dashboard: {str(e)}</p>",
                status_code=500,
            )

    async def _ensure_default_session(self) -> str:
        """
        Ensure default dashboard session exists and return its session_id.
        
        This method creates a default session with consistent task_id "dashboard"
        if it doesn't exist, or returns the existing session_id.
        
        Returns:
            Default dashboard session_id
        """
        # If we already have a cached session_id, check if it still exists
        if self._default_session_id:
            session = await self.session_middleware.get_session(self._default_session_id)
            if session is not None:
                return self._default_session_id
            # Session expired or not found, clear cache
            self._default_session_id = None
        
        # Generate session_id from default task_id
        session_id = self.session_middleware.generate_session_id(
            self.default_task_id,
            self.session_middleware.secret_key
        )
        
        # Check if session already exists
        session = await self.session_middleware.get_session(session_id)
        if session is not None:
            self._default_session_id = session_id
            return session_id
        
        # Create new session with default task_id
        try:
            from src.config import get_settings
            settings = get_settings()
            workspace_root = Path(settings.workspace.path).resolve()
            
            session = await self.session_middleware.create_session(
                task_id=self.default_task_id,
                session_id=session_id,
                metadata={
                    "created_via": "dashboard",
                    "workspace_mode": "server-side",
                    "is_default": True,
                },
                workspace_root=str(workspace_root),
            )
            self._default_session_id = session.id
            logger.info(f"Created default dashboard session: {session.id} (task_id: {self.default_task_id})")
            return session.id
        except Exception as e:
            logger.error(f"Failed to create default dashboard session: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to create default dashboard session: {str(e)}")

    async def _ensure_client(self, session_id: Optional[str] = None):
        """
        Ensure MCP client is connected for a specific session.
        
        Args:
            session_id: Optional session ID. If provided, creates/retrieves client for that session.
                       If None, creates a default client without session.
        """
        client_key = session_id or "default"
        
        if client_key not in self._clients or client_key not in self._transports:
            # Retry logic with exponential backoff
            max_retries = 3
            retry_delay = 0.5  # Start with 0.5 seconds
            
            for attempt in range(max_retries):
                try:
                    # Wait before retry (except first attempt)
                    if attempt > 0:
                        await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
                    
                    # Suppress deprecation warning for StreamableHttpTransport
                    # This is a known issue with fastmcp library versions
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        # Build headers with API key and optional session_id
                        headers = {"X-MCP-Key": self.api_key}
                        if session_id:
                            headers["X-Session-ID"] = session_id
                        
                        transport = StreamableHttpTransport(
                            url=self.server_url,
                            headers=headers,
                        )
                        client = Client(transport=transport)
                        await client.__aenter__()
                        
                        self._transports[client_key] = transport
                        self._clients[client_key] = client
                        logger.info(f"Successfully connected MCP client for session: {client_key}")
                        break  # Success, exit retry loop
                except httpx.HTTPStatusError as e:
                    # Handle HTTP errors (like 502 Bad Gateway)
                    if e.response.status_code == 502:
                        error_msg = "MCP server returned 502 Bad Gateway. Server may not be fully started yet."
                        if attempt < max_retries - 1:
                            logger.warning(f"{error_msg} Retrying in {retry_delay * (2 ** attempt)}s... (attempt {attempt + 1}/{max_retries})")
                            continue
                        else:
                            logger.error(f"{error_msg} All retries exhausted.")
                            raise HTTPException(
                                status_code=503,
                                detail=f"MCP server is not ready. Please ensure the server is fully started and try again. Error: {str(e)}"
                            )
                    else:
                        logger.error(f"HTTP error connecting to MCP server: {e}", exc_info=True)
                        raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {str(e)}")
                except httpx.ConnectError as e:
                    # Handle connection errors
                    error_msg = f"Cannot connect to MCP server at {self.server_url}. Server may not be running."
                    if attempt < max_retries - 1:
                        logger.warning(f"{error_msg} Retrying in {retry_delay * (2 ** attempt)}s... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error(f"{error_msg} All retries exhausted.")
                        raise HTTPException(
                            status_code=503,
                            detail=f"Cannot connect to MCP server at {self.server_url}. Please ensure the server is running. Error: {str(e)}"
                        )
                except Exception as e:
                    # Handle other errors
                    error_msg = f"Failed to create MCP client: {str(e)}"
                    if attempt < max_retries - 1:
                        logger.warning(f"{error_msg} Retrying in {retry_delay * (2 ** attempt)}s... (attempt {attempt + 1}/{max_retries})")
                        continue
                    else:
                        logger.error(f"{error_msg} All retries exhausted.", exc_info=True)
                        raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {error_msg}")
            else:
                # All retries exhausted
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to connect to MCP server after {max_retries} attempts. Please check if the server is running at {self.server_url}"
                )
        
        return self._clients[client_key]

    async def _get_tools(self) -> Dict[str, Any]:
        """Get all available tools"""
        try:
            client = await self._ensure_client()
            tools = await client.list_tools()
            
            tools_list = []
            for tool in tools:
                try:
                    # FastMCP tool objects have name, description, and inputSchema attributes
                    tool_name = getattr(tool, 'name', None)
                    tool_description = getattr(tool, 'description', None) or ""
                    tool_input_schema = getattr(tool, 'inputSchema', None) or {}
                    
                    # Fallback: if tool is a dict-like object
                    if tool_name is None and hasattr(tool, '__dict__'):
                        tool_dict = tool.__dict__
                        tool_name = tool_dict.get('name') or tool_dict.get('tool_name')
                        tool_description = tool_dict.get('description', tool_description)
                        tool_input_schema = tool_dict.get('inputSchema', tool_dict.get('input_schema', tool_input_schema))
                    
                    # Final fallback
                    if tool_name is None:
                        tool_name = str(tool) if tool else "unknown"
                    
                    tools_list.append({
                        "name": tool_name,
                        "description": tool_description,
                        "inputSchema": tool_input_schema,
                    })
                except Exception as tool_error:
                    logger.warning(f"Failed to process tool: {tool_error}, tool object: {tool}")
                    continue

            return {"tools": tools_list, "count": len(tools_list)}
        except Exception as e:
            logger.error(f"Failed to get tools: {e}", exc_info=True)
            # Return empty list instead of raising error to allow dashboard to load
            return {"tools": [], "count": 0, "error": str(e)}

    def _aggregate_session_stats_from_jsonl(self, log_path: Path) -> Dict[str, Dict[str, Any]]:
        """Parse JSONL tool-call log; return session_id -> stats."""
        session_stats: Dict[str, Dict[str, Any]] = {}
        if not log_path.exists():
            return session_stats
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    session_id = entry.get("session_id")
                    if not session_id:
                        continue
                    ts = entry.get("timestamp")
                    if session_id not in session_stats:
                        session_stats[session_id] = {
                            "tool_calls": 0,
                            "last_accessed": ts,
                            "first_seen": ts,
                        }
                    session_stats[session_id]["tool_calls"] += 1
                    if ts:
                        if (
                            not session_stats[session_id]["last_accessed"]
                            or ts > session_stats[session_id]["last_accessed"]
                        ):
                            session_stats[session_id]["last_accessed"] = ts
                        if (
                            not session_stats[session_id]["first_seen"]
                            or ts < session_stats[session_id]["first_seen"]
                        ):
                            session_stats[session_id]["first_seen"] = ts
        except Exception as e:
            logger.warning(f"Failed to read log file {log_path}: {e}")
        return session_stats

    def _merge_session_stats(self, *stats_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for d in stats_dicts:
            for sid, v in d.items():
                if sid not in merged:
                    merged[sid] = {
                        "tool_calls": v.get("tool_calls", 0),
                        "last_accessed": v.get("last_accessed"),
                        "first_seen": v.get("first_seen"),
                    }
                else:
                    merged[sid]["tool_calls"] += v.get("tool_calls", 0)
                    la = v.get("last_accessed")
                    fs = v.get("first_seen")
                    if la and (
                        not merged[sid]["last_accessed"] or la > merged[sid]["last_accessed"]
                    ):
                        merged[sid]["last_accessed"] = la
                    if fs and (
                        not merged[sid]["first_seen"] or fs < merged[sid]["first_seen"]
                    ):
                        merged[sid]["first_seen"] = fs
        return merged

    async def _get_sessions(self) -> Dict[str, Any]:
        """Get active sessions plus historical sessions found in tool-call logs."""
        try:
            sessions = await self.session_middleware.list_sessions()

            primary = self._aggregate_session_stats_from_jsonl(Path(self.tool_calls_log_file))
            fallback = (
                self._aggregate_session_stats_from_jsonl(Path(self.fallback_log_file))
                if self.fallback_log_file
                else {}
            )
            session_stats = self._merge_session_stats(primary, fallback)

            active_ids = {s.id for s in sessions}
            sessions_data: List[Dict[str, Any]] = []

            ws_root = Path(self.workspace_root).resolve()
            for session in sessions:
                stats = session_stats.get(session.id, {})
                tool_calls_log: Optional[str] = None
                if session.workspace_path:
                    try:
                        wp = Path(session.workspace_path).resolve()
                        tcp = wp / "tool_call" / "tool_calls.jsonl"
                        if tcp.is_file():
                            tool_calls_log = str(tcp.relative_to(ws_root)).replace("\\", "/")
                    except Exception:
                        tool_calls_log = None
                sessions_data.append({
                    "session_id": session.id,
                    "task_id": session.task_id,
                    "workspace_path": str(session.workspace_path) if session.workspace_path else None,
                    "tool_calls_log": tool_calls_log,
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "tool_calls": stats.get("tool_calls", 0),
                    "active": True,
                })

            for sid, stats in session_stats.items():
                if sid in active_ids:
                    continue
                task_guess = sid.rsplit("_", 1)[0] if "_" in sid else sid
                sessions_data.append({
                    "session_id": sid,
                    "task_id": task_guess,
                    "workspace_path": None,
                    "tool_calls_log": None,
                    "created_at": stats.get("first_seen") or stats.get("last_accessed"),
                    "last_accessed": stats.get("last_accessed"),
                    "tool_calls": stats.get("tool_calls", 0),
                    "active": False,
                })

            # Active sessions first, then by last_accessed descending (ISO timestamps)
            sessions_data.sort(
                key=lambda x: (1 if x.get("active") else 0, x.get("last_accessed") or ""),
                reverse=True,
            )

            return {
                "sessions": sessions_data,
                "count": len(sessions_data),
            }
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session"""
        try:
            deleted = await self.session_middleware.delete_session(session_id)
            if deleted:
                return {"success": True, "message": f"Session {session_id} deleted successfully"}
            else:
                raise HTTPException(
                    status_code=404, detail=f"Session {session_id} not found"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete session: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def _entry_to_call_dict(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        result = entry.get("result")
        if isinstance(result, str) and result.startswith("<") and "object at 0x" in result:
            result = None
        return {
            "timestamp": entry.get("timestamp"),
            "tool_name": entry.get("tool_name"),
            "arguments": entry.get("arguments", {}),
            "success": entry.get("success", True),
            "duration_ms": entry.get("duration_ms", 0),
            "result": result,
            "error": entry.get("error"),
            "session_id": entry.get("session_id"),
        }

    def _load_calls_from_jsonl_files(
        self, paths: List[Path], session_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for log_path in paths:
            if not log_path.exists():
                continue
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if session_id and entry.get("session_id") != session_id:
                            continue
                        calls.append(self._entry_to_call_dict(entry))
            except Exception as e:
                logger.warning(f"Failed to read log file {log_path}: {e}")
        calls.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return calls[:limit]

    async def _get_history(
        self,
        session_id: Optional[str],
        workspace_log: Optional[str],
        limit: int,
    ) -> Dict[str, Any]:
        """History from server JSONL (by session_id) or from workspace tool_call/tool_calls.jsonl."""
        try:
            if workspace_log:
                ws = Path(self.workspace_root).resolve()
                rel = workspace_log.strip().replace("\\", "/").lstrip("/")
                if ".." in rel or rel.startswith("/"):
                    raise HTTPException(status_code=400, detail="Invalid workspace_log path")
                full = (ws / rel).resolve()
                try:
                    full.relative_to(ws)
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="workspace_log must be under server workspace root"
                    )
                if full.name != "tool_calls.jsonl" or "tool_call" not in full.parts:
                    raise HTTPException(
                        status_code=400,
                        detail="workspace_log must end with tool_call/tool_calls.jsonl",
                    )
                calls = self._load_calls_from_jsonl_files([full], session_id=None, limit=limit)
                return {
                    "session_id": None,
                    "workspace_log": rel,
                    "calls": calls,
                    "count": len(calls),
                    "source": "workspace_file",
                }

            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="Provide session_id or workspace_log query parameter",
                )

            paths = [Path(self.tool_calls_log_file)]
            if self.fallback_log_file:
                fb = Path(self.fallback_log_file)
                if fb.resolve() != paths[0].resolve():
                    paths.append(fb)
            calls = self._load_calls_from_jsonl_files(paths, session_id=session_id, limit=limit)
            return {
                "session_id": session_id,
                "calls": calls,
                "count": len(calls),
                "source": "server_log",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get history: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    def _walk_tool_calls_logs_under(self, root: Path) -> List[Path]:
        """Return tool_call/tool_calls.jsonl paths under *root* using pruned os.walk."""
        found: List[Path] = []
        if not root.is_dir():
            return found
        try:
            for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                dirnames[:] = [
                    d
                    for d in dirnames
                    if d not in _TOOL_TRACE_PRUNE_SUBDIRS and not d.startswith(".")
                ]
                here = Path(dirpath)
                if here.name == "tool_call" and "tool_calls.jsonl" in filenames:
                    found.append(here / "tool_calls.jsonl")
                    dirnames.clear()
        except OSError as e:
            logger.debug("walk_tool_calls_logs_under skip %s: %s", root, e)
        return found

    async def _get_workspace_traces(self) -> Dict[str, Any]:
        """Discover per-workspace tool_calls.jsonl files (MAS worker / step dirs)."""
        try:
            ws = Path(self.workspace_root).resolve()
            if not ws.is_dir():
                return {"traces": [], "count": 0, "workspace_root": str(ws)}

            traces: List[Dict[str, Any]] = []
            max_files = 1200
            seen: set[str] = set()

            # Avoid Path.rglob on the whole tree (can take minutes). Prefer newest
            # top-level run directories first, prune heavy subtrees, stop early.
            try:
                subdirs = [p for p in ws.iterdir() if p.is_dir()]
            except OSError:
                subdirs = []

            def _mtime(p: Path) -> float:
                try:
                    return p.stat().st_mtime
                except OSError:
                    return 0.0

            subdirs.sort(key=_mtime, reverse=True)

            def _append_trace(p: Path) -> None:
                nonlocal traces
                try:
                    rel = p.relative_to(ws)
                except ValueError:
                    return
                rel_s = str(rel).replace("\\", "/")
                if rel_s in seen:
                    return
                seen.add(rel_s)
                n_lines = 0
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                n_lines += 1
                except OSError:
                    n_lines = -1
                parent = p.parent.parent
                traces.append(
                    {
                        "relative_path": rel_s,
                        "call_count": n_lines,
                        "label": str(parent.name) if parent.parts else rel_s,
                        "parent_hint": str(parent.relative_to(ws)).replace("\\", "/")
                        if parent != ws
                        else "",
                    }
                )

            for sd in subdirs:
                if len(traces) >= max_files:
                    break
                for p in self._walk_tool_calls_logs_under(sd):
                    _append_trace(p)
                    if len(traces) >= max_files:
                        break

            # Root-level tool_call/tool_calls.jsonl (not under a child of ws)
            if len(traces) < max_files:
                root_tcp = ws / "tool_call" / "tool_calls.jsonl"
                if root_tcp.is_file():
                    _append_trace(root_tcp)

            traces.sort(key=lambda t: t.get("relative_path", ""), reverse=True)
            return {
                "traces": traces,
                "count": len(traces),
                "workspace_root": str(ws),
            }
        except Exception as e:
            logger.error(f"Failed to scan workspace traces: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def _execute_tool(self, request: Request) -> Dict[str, Any]:
        """Execute a tool from the dashboard"""
        try:
            body = await request.json()
            tool_name = body.get("tool_name")
            tool_arguments = body.get("arguments", {})

            if not tool_name:
                raise HTTPException(status_code=400, detail="tool_name is required")

            # Get or create dashboard session
            # Use default dashboard session (consistent task_id "dashboard")
            dashboard_session_id = request.headers.get("X-Session-ID")
            if not dashboard_session_id:
                # Ensure default dashboard session exists (with consistent task_id "dashboard")
                dashboard_session_id = await self._ensure_default_session()

            # Execute tool via MCP client with session_id in headers
            try:
                # Get or create client for this session (with session_id in headers)
                client = await self._ensure_client(session_id=dashboard_session_id)
                # Call tool WITHOUT meta parameter - session_id is in HTTP headers
                result = await client.call_tool(
                    tool_name,
                    tool_arguments,
                )
                
                # Extract result from MCP response
                # FastMCP returns result with .data attribute and .is_error flag
                if hasattr(result, 'is_error') and result.is_error:
                    # Error response
                    error_msg = "Tool execution failed"
                    if hasattr(result, 'content') and result.content:
                        error_msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    return {
                        "success": False,
                        "error": error_msg,
                        "data": None,
                        "session_id": dashboard_session_id,
                    }
                elif hasattr(result, 'data'):
                    # Success response with .data attribute
                    result_data = result.data
                    if isinstance(result_data, dict):
                        return {
                            "success": result_data.get("success", True),
                            "data": result_data.get("data"),
                            "error": result_data.get("error"),
                            "session_id": dashboard_session_id,
                        }
                    else:
                        return {
                            "success": True,
                            "data": result_data,
                            "session_id": dashboard_session_id,
                        }
                elif hasattr(result, "content") and result.content:
                    # MCP content format
                    content_text = result.content[0].text if result.content else "{}"
                    try:
                        result_data = json.loads(content_text)
                        return {
                            "success": result_data.get("success", True),
                            "data": result_data.get("data"),
                            "error": result_data.get("error"),
                            "session_id": dashboard_session_id,
                        }
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "data": content_text,
                            "session_id": dashboard_session_id,
                        }
                elif isinstance(result, dict):
                    return {
                        "success": result.get("success", True),
                        "data": result.get("data"),
                        "error": result.get("error"),
                        "session_id": dashboard_session_id,
                    }
                else:
                    return {
                        "success": True,
                        "data": result,
                        "session_id": dashboard_session_id,
                    }
            except Exception as e:
                logger.error(f"Tool execution error: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": dashboard_session_id,
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to execute tool: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def on_request(
        self, context: MiddlewareContext, call_next
    ) -> any:
        """
        Dashboard middleware hook.
        
        Note: Dashboard routes are handled via FastAPI routes registered
        by register_routes(), not through this middleware hook.
        This hook is kept for consistency with other middlewares but
        simply passes through all requests.
        """
        return await call_next(context)
