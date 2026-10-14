"""MCP client, wrapping fastmcp"""
import asyncio
import logging
import os
import secrets
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from ..utils.exceptions import MCPError

logger = logging.getLogger(__name__)


def _ensure_no_proxy_for_local(server_url: str) -> None:
    """
    When server_url points to localhost, add 127.0.0.1/localhost to NO_PROXY
    to prevent local MCP requests from being incorrectly routed through a
    VPN/system proxy. Also sets no_proxy for compatibility with some environments
    (e.g. Windows).
    """
    try:
        host = urlparse(server_url).hostname
        if host in ("127.0.0.1", "localhost", "::1", None):
            current = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").strip()
            add = "127.0.0.1,localhost,::1"
            if not current:
                val = add
            elif "127.0.0.1" not in current and "localhost" not in current:
                val = f"{add},{current}"
            else:
                return
            os.environ["NO_PROXY"] = val
            os.environ["no_proxy"] = val
    except Exception:
        pass


_SESSION_ERROR_MARKERS = (
    "Session validation failed",
    "Session not found",
    "Missing session ID",
)

class MCPClient:
    """MCP client, wrapping fastmcp

    Each MCPClient instance automatically appends a short random hex suffix to
    the user-supplied ``task_id`` before creating a session on the MCP server.
    This guarantees that concurrent experiments sharing the same logical task
    name (e.g. "optimizer" or a problem_id) each get an isolated server-side
    session — closing one will never affect the other.

    The original name is accessible via :attr:`logical_task_id`; the effective
    (suffixed) name used for the handshake is :attr:`task_id`.
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: str,
        task_id: str,
        workspace_subdir: Optional[str] = None,
        max_session_retries: int = 2,
        skip_session_dir: bool = False,
    ):
        """
        Initialize the MCP client.

        A random 16-hex-char instance suffix is appended to task_id so that
        multiple MCPClient instances with the same logical task_id will each
        get their own independent server session.
        """
        self.server_url = server_url
        self.api_key = api_key

        self._instance_suffix = secrets.token_hex(8)
        self.logical_task_id = task_id
        self.task_id = f"{task_id}-{self._instance_suffix}"

        self.workspace_subdir = workspace_subdir
        self.skip_session_dir = skip_session_dir
        self._client: Optional[Client] = None
        self._transport: Optional[StreamableHttpTransport] = None
        self._session_id: Optional[str] = None
        self._workspace_path: Optional[str] = None  # workspace path returned from handshake (read-only property)
        self._connected = False
        self.max_session_retries = max_session_retries
        self._workbench_actions: List[str] = []

        logger.debug(
            "MCPClient created: logical_task_id=%s, effective_task_id=%s",
            self.logical_task_id, self.task_id,
        )

    @property
    def local_task_id(self) -> str:
        """The original task_id passed by the caller (without instance suffix)."""
        return self.logical_task_id

    @property
    def instance_suffix(self) -> str:
        """The 16-char hex suffix that makes this instance's task_id unique."""
        return self._instance_suffix

    def _sync_transport_session_header(self) -> None:
        """Attach ``X-Session-ID`` to the Streamable HTTP transport so server tools can
        resolve the workspace when ``call_tool`` meta is not visible on ``get_context()``.
        """
        if not self._session_id or self._transport is None:
            return
        try:
            h = getattr(self._transport, "headers", None)
            if isinstance(h, dict):
                merged = dict(h)
                merged["X-Session-ID"] = str(self._session_id)
                self._transport.headers = merged
        except Exception:
            logger.debug("Could not set X-Session-ID on MCP transport", exc_info=True)

    async def __aenter__(self):
        """Async context manager entry: establish connection and perform handshake"""
        # Bypass system proxy for local MCP access to avoid VPN connectivity issues
        _ensure_no_proxy_for_local(self.server_url)
        # Create transport
        self._transport = StreamableHttpTransport(
            url=self.server_url,
            headers={"X-MCP-Key": self.api_key},
        )
        
        # Create client
        self._client = Client(transport=self._transport)
        
        # Establish connection
        await self._client.__aenter__()
        self._connected = True
        
        # Perform handshake
        try:
            await self._handshake()
        except BaseException:
            await self._client.__aexit__(None, None, None)
            self._connected = False
            raise
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit: notify server to close session, then close connection"""
        if self._client is not None:
            if self._connected and self._session_id:
                try:
                    await asyncio.wait_for(
                        self._client.call_tool(
                            "close_session",
                            {},
                            meta={"session_id": self._session_id},
                        ),
                        timeout=3.0,
                    )
                except Exception:
                    pass
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._connected = False
            self._session_id = None
            self._workspace_path = None
    
    async def _handshake(self) -> None:
        """
        Perform handshake to establish a session.
        
        Supports specifying a workspace subdirectory (server-side managed):
        - If workspace_subdir is provided, it is passed to the handshake tool
        - If not provided, the server-side default workspace is used
        
        Raises:
            MCPError: If the handshake fails
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async with MCPClient(...)")
        
        handshake_params = {
            "task_id": self.task_id,
            "logical_task_id": self.logical_task_id,
            "skip_session_dir": self.skip_session_dir,
        }
        if self.workspace_subdir is not None:
            handshake_params["workspace_subdir"] = str(self.workspace_subdir)
        result = await self._client.call_tool("handshake", handshake_params)
        
        if result.is_error:
            error_msg = "Handshake failed"
            if result.content and len(result.content) > 0:
                error_msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            raise MCPError(f"Handshake failed: {error_msg}")

        raw_data = getattr(result, 'data', None)
        if isinstance(raw_data, dict) and raw_data.get("success") is not None:
            payload = raw_data.get("data") if raw_data.get("success") else None
            if not raw_data.get("success"):
                raise MCPError(f"Handshake failed: {raw_data.get('error') or 'unknown error'}")
            if not isinstance(payload, dict):
                raise MCPError("Handshake succeeded but invalid response format")
            self._session_id = payload.get('session_id')
            if not self._session_id:
                raise MCPError("Handshake succeeded but session_id not found in response")
            self._workspace_path = payload.get('workspace_path')
            self._sync_transport_session_header()
            return
        
        # Backward-compatible fallback for older server responses
        if isinstance(raw_data, dict):
            self._session_id = raw_data.get('session_id')
            if not self._session_id:
                raise MCPError("Handshake succeeded but session_id not found in response")
            self._workspace_path = raw_data.get('workspace_path')
            self._sync_transport_session_header()
            return

        raise MCPError("Handshake succeeded but invalid response format")
    

    # ------------------------------------------------------------------
    # Session-error auto-reconnect
    # ------------------------------------------------------------------

    @staticmethod
    def _is_session_error(error_msg: str) -> bool:
        """Check if an error message indicates a session-related failure."""
        if not error_msg:
            return False
        return any(marker in error_msg for marker in _SESSION_ERROR_MARKERS)

    async def _reconnect(self) -> bool:
        """Re-establish connection and session via handshake.

        Returns True on success, False on failure.
        """
        try:
            logger.warning("Attempting session reconnect for task_id=%s", self.task_id)
            if self._client is not None:
                try:
                    await self._client.__aexit__(None, None, None)
                except Exception:
                    pass

            _ensure_no_proxy_for_local(self.server_url)
            self._transport = StreamableHttpTransport(
                url=self.server_url,
                headers={"X-MCP-Key": self.api_key},
            )
            self._client = Client(transport=self._transport)
            await self._client.__aenter__()
            self._connected = True
            await self._handshake()
            logger.info("Session reconnect succeeded: session_id=%s", self._session_id)
            return True
        except Exception as e:
            logger.error("Session reconnect failed: %s", e)
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> Optional[str]:
        """
        Get the current session ID.
        
        Returns:
            The session ID, or None if not connected
        """
        return self._session_id
    
    def clear_workbench_actions(self) -> None:
        """Clear client-side mirror of WorkBench actions (server is authoritative)."""
        self._workbench_actions = []

    def get_workbench_actions(self) -> List[str]:
        return list(self._workbench_actions)

    @property
    def workspace_path(self) -> Optional[str]:
        """
        Get the current workspace relative path (relative to server workspace base root).
        
        Returns:
            The workspace relative path, or None if not connected or workspace not created
        """
        return self._workspace_path
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.
        
        Returns:
            List of tools, each containing name, description, etc.
        
        Raises:
            RuntimeError: If the client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with MCPClient(...)")
        
        tools = await self._client.list_tools()
        
        # Convert to list of dicts
        result = []
        for tool in tools:
            result.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            })
        
        return result
    
    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema of a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            The tool's JSON Schema, or None if the tool does not exist
        
        Raises:
            RuntimeError: If the client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with MCPClient(...)")
        
        tools = await self.list_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return tool.get("inputSchema")
        
        return None
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float = 900.0,
        timing_detail: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Call a tool.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Timeout in seconds, defaults to 900s (15 minutes)
        
        Returns:
            Tool execution result in the format:
            {
                "success": bool,
                "data": Any,        # None on failure
                "error": Optional[str]  # None on success
            }
        
        Raises:
            RuntimeError: If the client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with MCPClient(...)")
        
        if not self._session_id:
            raise RuntimeError("Session not established. Handshake may have failed.")

        def _mark(name: str, t0: float) -> None:
            if timing_detail is not None:
                timing_detail[name] = round(time.perf_counter() - t0, 6)

        session_retries = 0
        while True:
            t_all = time.perf_counter()
            result_dict = await self._call_tool_once(tool_name, arguments, timeout, timing_detail, t_all, _mark)

            if result_dict.get("success") or session_retries >= self.max_session_retries:
                return result_dict

            error_msg = result_dict.get("error", "")
            if not self._is_session_error(error_msg):
                return result_dict

            session_retries += 1
            logger.warning(
                "Session error on tool '%s' (attempt %d/%d): %s — reconnecting",
                tool_name, session_retries, self.max_session_retries, error_msg,
            )
            reconnected = await self._reconnect()
            if not reconnected:
                result_dict["error"] = f"{error_msg} (reconnect also failed after {session_retries} attempts)"
                return result_dict


    async def _call_tool_once(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: float,
        timing_detail: Optional[Dict[str, float]],
        t_all: float,
        _mark,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call attempt.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            timeout: Timeout in seconds, defaults to 900s (15 minutes)
        
        Returns:
            Tool execution result in the format:
            {
                "success": bool,
                "data": Any,        # None on failure
                "error": Optional[str]  # None on success
            }
        
        Raises:
            RuntimeError: If the client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with MCPClient(...)")
        
        if not self._session_id:
            raise RuntimeError("Session not established. Handshake may have failed.")

        def _mark(name: str, t0: float) -> None:
            if timing_detail is not None:
                timing_detail[name] = round(time.perf_counter() - t0, 6)

        t_all = time.perf_counter()
        try:
            t_rpc = time.perf_counter()
            result = await asyncio.wait_for(
                self._client.call_tool(
                    tool_name,
                    arguments,
                    meta={"session_id": self._session_id}
                ),
                timeout=timeout
            )
            _mark("mcp_transport_rpc_seconds", t_rpc)

            # Normalise: guarantee the canonical {success, data, error} shape
            # regardless of what the server returns in result.data.
            t_norm = time.perf_counter()
            raw = result.data
            if isinstance(raw, dict) and "success" in raw:
                out: Dict[str, Any] = dict(raw)
                if "data" not in out:
                    out["data"] = None
                if "error" not in out:
                    out["error"] = None if out.get("success") else "Unknown error"
                data = out.get("data")
                if isinstance(data, dict):
                    wb_action = data.get("workbench_action")
                    if wb_action:
                        self._workbench_actions.append(str(wb_action))
                _mark("normalize_result_seconds", t_norm)
                _mark("total_call_tool_seconds", t_all)
                return out
            # Server returned something unexpected — wrap it
            if result.is_error:
                text = ""
                if result.content:
                    text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
                _mark("normalize_result_seconds", t_norm)
                _mark("total_call_tool_seconds", t_all)
                return {"success": False, "data": None, "error": text or "Tool call failed (server error)"}
            _mark("normalize_result_seconds", t_norm)
            _mark("total_call_tool_seconds", t_all)
            return {"success": True, "data": raw, "error": None}

        except asyncio.TimeoutError:
            self._connected = False
            _mark("total_call_tool_seconds", t_all)
            return {
                "success": False,
                "data": None,
                "error": f"Tool call timed out after {timeout}s. MCP connection may have been lost."
            }
        except (ConnectionError, OSError) as e:
            self._connected = False
            _mark("total_call_tool_seconds", t_all)
            return {
                "success": False,
                "data": None,
                "error": f"MCP connection error: {e}. Connection may have been lost."
            }
        except Exception as e:
            err_msg = str(e)
            if "ValidationError" in type(e).__name__ or "json_invalid" in err_msg or "EOF while parsing" in err_msg:
                self._connected = False
                err_msg = (
                    f"MCP response invalid or truncated (tool output may be too large). "
                    f"Original: {err_msg}"
                )
            _mark("total_call_tool_seconds", t_all)
            return {
                "success": False,
                "data": None,
                "error": f"Tool call exception: {err_msg}"
            }

