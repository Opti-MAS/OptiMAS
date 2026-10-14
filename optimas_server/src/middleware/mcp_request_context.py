"""Per-request MCP session id for tools.

``SessionMiddleware`` validates ``session_id`` and sets state on ``MiddlewareContext``.
Tools that call ``get_context()`` often see a different context without that state.

We mirror the validated session id in two ways (either may work depending on FastMCP):

1. **asyncio.Task–keyed map** — ``await call_next`` usually runs the tool in the same
   ``asyncio.Task`` as the middleware coroutine; mapping ``id(current_task())`` is
   reliable when ``ContextVar`` does not propagate across FastMCP internals.
2. **ContextVar** — kept as a secondary path for environments where it does propagate.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Dict, Optional

# --- ContextVar (optional propagation) ---
_mcp_session_id: ContextVar[Optional[str]] = ContextVar("mcp_session_id", default=None)

# --- Same-Task map (primary for FastMCP tool execution) ---
_task_session_ids: Dict[int, str] = {}


def get_tool_session_id_for_task() -> Optional[str]:
    """Session id bound to the current asyncio Task (set by SessionMiddleware)."""
    task = asyncio.current_task()
    if task is None:
        return None
    sid = _task_session_ids.get(id(task))
    if sid is None or sid == "":
        return None
    return str(sid).strip()


def attach_tool_session_to_current_task(session_id: str) -> Optional[int]:
    """Bind *session_id* to ``asyncio.current_task()``; returns task id for ``clear_tool_session_for_task``."""
    task = asyncio.current_task()
    if task is None:
        return None
    key = id(task)
    _task_session_ids[key] = str(session_id).strip()
    return key


def clear_tool_session_for_task(task_key: Optional[int]) -> None:
    if task_key is not None:
        _task_session_ids.pop(task_key, None)


def get_tool_session_id() -> Optional[str]:
    """Session id from ContextVar (if set for this logical context)."""
    v = _mcp_session_id.get()
    if v is None or v == "":
        return None
    return str(v).strip()


def get_mcp_tool_session_id_any() -> Optional[str]:
    """Prefer Task map, then ContextVar."""
    sid = get_tool_session_id_for_task()
    if sid:
        return sid
    return get_tool_session_id()


def push_tool_session_id(session_id: str) -> Token:
    return _mcp_session_id.set(str(session_id).strip())


def pop_tool_session_id(token: Token) -> None:
    _mcp_session_id.reset(token)
