"""Helpers for FastMCP context ``get_state`` which may be sync or async (awaitable)."""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    if value is None:
        return None
    if inspect.isawaitable(value) or inspect.iscoroutine(value):
        return await value
    return value


async def get_state_safe(context_obj: Any, key: str) -> Any:
    """Call ``context_obj.get_state(key)`` and await if the implementation returns a coroutine."""
    if context_obj is None:
        return None
    get_state = getattr(context_obj, "get_state", None)
    if not callable(get_state):
        return None
    try:
        raw = get_state(key)
    except Exception:
        return None
    return await maybe_await(raw)
