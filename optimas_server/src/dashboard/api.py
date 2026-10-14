"""
Dashboard API endpoints for OptiMAS Server.

Provides REST API endpoints for the Dashboard UI.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from src.middleware.session_middleware import SessionMiddleware
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardAPI:
    """Dashboard API endpoints"""

    def __init__(
        self,
        mcp_instance,
        session_middleware: SessionMiddleware,
        server_url: str,
        api_key: str,
        log_file: str = "logs/mcp_server_tools.jsonl",
    ):
        """
        Initialize Dashboard API.

        Args:
            mcp_instance: FastMCP instance (for accessing tools)
            session_middleware: SessionMiddleware instance
            server_url: MCP server URL (e.g., "http://127.0.0.1:8188/mcp")
            api_key: API key for authentication
            log_file: Path to log file for tool call history
        """
        self.mcp = mcp_instance
        self.session_middleware = session_middleware
        self.server_url = server_url
        self.api_key = api_key
        self.log_file = log_file
        self.router = APIRouter()

        # Register routes
        self._register_routes()
        
        # Create internal client for tool operations
        self._client: Optional[Client] = None
        self._transport: Optional[StreamableHttpTransport] = None

    def _register_routes(self):
        """Register all dashboard routes"""
        self.router.add_api_route("/api/tools", self.get_tools, methods=["GET"])
        self.router.add_api_route("/api/sessions", self.get_sessions, methods=["GET"])
        self.router.add_api_route(
            "/api/sessions/{session_id}", self.delete_session, methods=["DELETE"]
        )
        self.router.add_api_route("/api/history", self.get_history, methods=["GET"])
        self.router.add_api_route(
            "/api/execute-tool", self.execute_tool, methods=["POST"]
        )

    async def _ensure_client(self):
        """Ensure MCP client is connected"""
        if self._client is None or self._transport is None:
            try:
                self._transport = StreamableHttpTransport(
                    url=self.server_url,
                    headers={"X-MCP-Key": self.api_key},
                )
                self._client = Client(transport=self._transport)
                await self._client.__aenter__()
            except Exception as e:
                logger.error(f"Failed to create MCP client: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {str(e)}")
    
    async def get_tools(self) -> Dict[str, Any]:
        """Get all available tools"""
        try:
            await self._ensure_client()
            tools = await self._client.list_tools()
            
            tools_list = []
            for tool in tools:
                tools_list.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema or {},
                })

            return {"tools": tools_list, "count": len(tools_list)}
        except Exception as e:
            logger.error(f"Failed to get tools: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def get_sessions(self) -> Dict[str, Any]:
        """Get all active sessions"""
        try:
            sessions = await self.session_middleware.list_sessions()
            
            # Get tool call counts from log file
            log_path = Path(self.log_file)
            session_stats = {}
            
            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry = json.loads(line)
                                    session_id = entry.get("session_id")
                                    if session_id:
                                        if session_id not in session_stats:
                                            session_stats[session_id] = {
                                                "tool_calls": 0,
                                                "last_accessed": entry.get("timestamp"),
                                            }
                                        session_stats[session_id]["tool_calls"] += 1
                                        # Update last accessed if newer
                                        if entry.get("timestamp"):
                                            if (
                                                not session_stats[session_id]["last_accessed"]
                                                or entry["timestamp"]
                                                > session_stats[session_id]["last_accessed"]
                                            ):
                                                session_stats[session_id][
                                                    "last_accessed"
                                                ] = entry["timestamp"]
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to read log file: {e}")

            sessions_data = []
            for session in sessions:
                stats = session_stats.get(session.id, {})
                sessions_data.append({
                    "session_id": session.id,
                    "task_id": session.task_id,
                    "workspace_path": str(session.workspace_path) if session.workspace_path else None,
                    "created_at": session.created_at.isoformat(),
                    "last_accessed": session.last_accessed.isoformat(),
                    "tool_calls": stats.get("tool_calls", 0),
                })

            return {
                "sessions": sessions_data,
                "count": len(sessions_data),
            }
        except Exception as e:
            logger.error(f"Failed to get sessions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_session(self, session_id: str) -> Dict[str, Any]:
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

    async def get_history(
        self, session_id: str = Query(..., description="Session ID"), limit: int = Query(100, ge=1, le=1000)
    ) -> Dict[str, Any]:
        """Get tool call history for a session"""
        try:
            log_path = Path(self.log_file)
            calls = []

            if log_path.exists():
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry = json.loads(line)
                                    if entry.get("session_id") == session_id:
                                        # Extract result, filtering out invalid object string representations
                                        result = entry.get("result")
                                        # Filter out ToolResult object string representations
                                        if isinstance(result, str) and result.startswith("<") and "object at 0x" in result:
                                            result = None  # Invalid result, set to None
                                        
                                        # Map optimas_server log format to dashboard format
                                        calls.append({
                                            "timestamp": entry.get("timestamp"),
                                            "tool_name": entry.get("tool_name"),
                                            "arguments": entry.get("arguments", {}),
                                            "success": entry.get("success", True),
                                            "duration_ms": entry.get("duration_ms", 0),
                                            "result": result,  # May be string, dict, list, or None
                                            "error": entry.get("error"),
                                        })
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.warning(f"Failed to read log file: {e}")

            # Sort by timestamp (most recent first)
            calls.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            calls = calls[:limit]

            return {
                "session_id": session_id,
                "calls": calls,
                "count": len(calls),
            }
        except Exception as e:
            logger.error(f"Failed to get history: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def execute_tool(self, request: Request) -> Dict[str, Any]:
        """Execute a tool from the dashboard"""
        try:
            body = await request.json()
            tool_name = body.get("tool_name")
            tool_arguments = body.get("arguments", {})

            if not tool_name:
                raise HTTPException(status_code=400, detail="tool_name is required")

            # Get or create dashboard session
            dashboard_session_id = request.headers.get("X-Session-ID")
            if not dashboard_session_id:
                # Create a temporary dashboard session via handshake
                await self._ensure_client()
                try:
                    handshake_result = await self._client.call_tool(
                        "handshake",
                        {"task_id": "dashboard"},
                    )
                    # Extract session_id from result
                    if isinstance(handshake_result, dict):
                        dashboard_session_id = handshake_result.get("session_id")
                    elif hasattr(handshake_result, "content"):
                        # MCP result format
                        import json
                        content = handshake_result.content[0].text if handshake_result.content else "{}"
                        result_data = json.loads(content)
                        dashboard_session_id = result_data.get("session_id")
                    
                    if not dashboard_session_id:
                        raise HTTPException(status_code=500, detail="Failed to create dashboard session")
                    logger.info(f"Created dashboard session: {dashboard_session_id}")
                except Exception as e:
                    logger.error(f"Failed to create dashboard session: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

            # Execute tool via MCP client
            try:
                await self._ensure_client()
                result = await self._client.call_tool(
                    tool_name,
                    tool_arguments,
                    meta={"session_id": dashboard_session_id},
                )
                
                # Extract result from MCP response
                if hasattr(result, "content") and result.content:
                    import json
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
