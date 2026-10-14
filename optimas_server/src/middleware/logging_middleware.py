"""
Logging middleware for tool call logging.

Records tool calls to file and sends logs to client via Context.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp.server.middleware import Middleware, MiddlewareContext

from src.middleware.session_middleware import resolve_tool_call_session_id

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(Middleware):
    """
    Middleware for logging tool calls.

    Features:
    - Records tool calls to JSON log file
    - Sends logs to client via Context
    - Tracks execution time
    - Handles errors gracefully
    """

    def __init__(
        self,
        log_file: str = "logs/mcp_server_tools.jsonl",
        enable_file_logging: bool = True,
        enable_context_logging: bool = True,
        session_middleware: Optional[Any] = None,
    ):
        """
        Initialize logging middleware.

        Args:
            log_file: Path to log file (JSONL format)
            enable_file_logging: Enable file logging
            enable_context_logging: Enable Context logging to client
            session_middleware: Optional SessionMiddleware instance for accessing session workspace paths
        """
        self.log_file = log_file
        self.enable_file_logging = enable_file_logging
        self.enable_context_logging = enable_context_logging
        self.session_middleware = session_middleware

        # Ensure log directory exists
        if self.enable_file_logging:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

    async def on_call_tool(
        self, context: MiddlewareContext, call_next
    ) -> Any:
        """Log tool call before and after execution."""
        tool_name = context.message.name
        
        # Skip logging for handshake tool
        if tool_name == 'handshake':
            return await call_next(context)

        mw_t0 = time.perf_counter()
        start_time = time.time()
        arguments = context.message.arguments

        t_after_resolve = time.perf_counter()

        try:
            t_before_next = time.perf_counter()
            result = await call_next(context)
            t_after_next = time.perf_counter()
            duration_ms = (time.time() - start_time) * 1000

            # After ``call_next``, SessionMiddleware's ``finally`` may clear task-local session
            # mirrors and context state may be reset — but ``meta.session_id`` on this tool's
            # ``MiddlewareContext`` still resolves (see ``resolve_tool_call_session_id``).
            session_id = await resolve_tool_call_session_id(context)

            timing_phases_seconds = {
                "middleware_resolve_session_seconds": round(t_after_resolve - mw_t0, 6),
                "call_next_tool_chain_seconds": round(t_after_next - t_before_next, 6),
            }

            # Log success
            await self._log_tool_call(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                success=True,
                duration_ms=duration_ms,
                error=None,
                timing_phases_seconds=timing_phases_seconds,
                middleware_perf_t0=mw_t0,
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            t_err = time.perf_counter()
            session_id = await resolve_tool_call_session_id(context)
            timing_phases_seconds = {
                "middleware_resolve_session_seconds": round(t_after_resolve - mw_t0, 6),
                "middleware_until_exception_seconds": round(t_err - mw_t0, 6),
            }

            # Log error
            await self._log_tool_call(
                session_id=session_id,
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                success=False,
                duration_ms=duration_ms,
                error=str(e),
                timing_phases_seconds=timing_phases_seconds,
                middleware_perf_t0=mw_t0,
            )

            # Re-raise exception
            raise

    def _write_to_session_workspace(self, workspace_path: Path, log_entry: Dict):
        """
        Write log entry to session workspace's tool_call directory.
        
        Args:
            workspace_path: Session workspace path
            log_entry: Log entry dictionary
        """
        try:
            # Create tool_call directory in workspace
            tool_call_dir = workspace_path / "tool_call"
            tool_call_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to tool_call log file (JSONL format)
            tool_call_log_file = tool_call_dir / "tool_calls.jsonl"
            with open(tool_call_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write log entry to session workspace {workspace_path}: {e}")

    async def _resolve_task_id(self, session_id: Optional[str]) -> Optional[str]:
        """Resolve the logical task_id from the session for logging.

        The raw ``session_id`` includes an HMAC suffix and a per-client-instance
        random suffix (e.g. ``query-ab12_e28a``).  Downstream consumers such as
        the optimizer parse ``tool_calls.jsonl`` and may confuse these hashes
        with working-trajectory invocation IDs.

        Resolution order:
        1. ``session.metadata["logical_task_id"]`` — the original task name
           before any client-side uniqueness suffix (preferred).
        2. ``session.task_id`` — the effective task_id (may still contain the
           client instance suffix, but at least strips the HMAC hash).
        3. The raw ``session_id`` as fallback.
        """
        if not session_id or not self.session_middleware:
            return session_id
        try:
            session = await self.session_middleware.get_session(session_id)
            if session:
                logical = session.metadata.get("logical_task_id")
                if logical:
                    return logical
                if session.task_id:
                    return session.task_id
        except Exception:
            pass
        return session_id

    async def _log_tool_call(
        self,
        session_id: Optional[str],
        tool_name: str,
        arguments: Dict,
        result: Any,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        timing_phases_seconds: Optional[Dict[str, float]] = None,
        middleware_perf_t0: Optional[float] = None,
    ):
        """Log tool call to file."""
        t_fn0 = time.perf_counter()
        task_id = await self._resolve_task_id(session_id)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": task_id,
            "tool_name": tool_name,
            "arguments": self._sanitize_for_log(arguments),
            "success": success,
            "duration_ms": round(duration_ms, 2),
        }

        if error:
            log_entry["error"] = error

        t_extract0 = time.perf_counter()
        if result is not None:
            # Extract result content properly
            result_data = self._extract_result_content(result)
            if result_data is not None:
                log_entry["result"] = result_data
        extract_seconds = time.perf_counter() - t_extract0

        phases: Dict[str, float] = {**(timing_phases_seconds or {})}
        phases["log_extract_and_sanitize_result_seconds"] = round(extract_seconds, 6)

        t_write0 = time.perf_counter()
        # Write to file (JSONL format)
        if self.enable_file_logging:
            try:
                # Write to global log file
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write log entry to global log: {e}")
            
            # Also write to session workspace if available
            if session_id and self.session_middleware:
                try:
                    session = await self.session_middleware.get_session(session_id)
                    if session and session.workspace_path:
                        log_path = session.workspace_path
                        if session.metadata.get("log_base_path"):
                            log_path = Path(session.metadata["log_base_path"])
                            log_path.mkdir(parents=True, exist_ok=True)
                        self._write_to_session_workspace(log_path, log_entry)
                except Exception as e:
                    # If we can't get session, skip workspace logging (non-critical)
                    logger.debug(f"Could not write log entry to session workspace: {e}")

        phases["log_write_global_session_workspace_seconds"] = round(time.perf_counter() - t_write0, 6)
        phases["_log_tool_call_function_total_seconds"] = round(time.perf_counter() - t_fn0, 6)
        if middleware_perf_t0 is not None:
            phases["middleware_wall_since_on_call_tool_seconds"] = round(
                time.perf_counter() - middleware_perf_t0, 6
            )
        log_entry["timing_phases_seconds"] = phases

        # Also log to standard logger
        message = f"Tool call: {tool_name} (session={session_id}, duration={duration_ms:.2f}ms)"
        if error:
            message += f" - Error: {error}"

        if success:
            logger.info(message)
        else:
            logger.error(message)

    def _extract_result_content(self, result: Any) -> Any:
        """
        Extract content from tool result, handling ToolResult objects and other types.
        
        Args:
            result: Tool result (may be ToolResult object, dict, list, etc.)
        
        Returns:
            Extracted result content (dict, list, str, etc.) or None if extraction fails
        """
        try:
            # Handle ToolResult objects (from fastmcp)
            # ToolResult has a 'content' attribute which is a list of content items
            if hasattr(result, 'content') and result.content:
                # Extract text from content items
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_items.append(item.text)
                    elif isinstance(item, str):
                        content_items.append(item)
                    elif isinstance(item, dict):
                        content_items.append(item)
                    else:
                        content_items.append(str(item))
                
                # If single item, return it directly; otherwise return list
                if len(content_items) == 1:
                    extracted = content_items[0]
                else:
                    extracted = content_items
                
                # Try to parse as JSON if it's a string
                if isinstance(extracted, str):
                    try:
                        extracted = json.loads(extracted)
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep as string if not valid JSON
                
                return self._sanitize_for_log(extracted, max_depth=5)
            
            # Handle dict results (common format)
            elif isinstance(result, dict):
                return self._sanitize_for_log(result, max_depth=5)
            
            # Handle list results
            elif isinstance(result, list):
                return self._sanitize_for_log(result, max_depth=5)
            
            # Handle primitive types
            elif isinstance(result, (str, int, float, bool, type(None))):
                return result
            
            # For other types, try to convert to dict or string
            else:
                # Try to get dict representation if available
                if hasattr(result, '__dict__'):
                    try:
                        return self._sanitize_for_log(result.__dict__, max_depth=5)
                    except Exception:
                        pass
                
                # Fallback to string representation (truncated)
                result_str = str(result)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "... (truncated)"
                return result_str
                
        except Exception as e:
            logger.warning(f"Failed to extract result content: {e}")
            # Fallback to string representation
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"
            return result_str

    def _sanitize_for_log(self, data: Any, max_depth: int = 3) -> Any:
        """Sanitize data for logging (prevent circular references, limit depth)."""
        if max_depth <= 0:
            return "... (max depth)"

        if isinstance(data, dict):
            return {
                k: self._sanitize_for_log(v, max_depth - 1)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_for_log(item, max_depth - 1) for item in data[:10]]  # Limit list size
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)[:200]  # Truncate other types

