"""Register WorkBench sandbox tools on the MCP server."""
from __future__ import annotations

import asyncio
import inspect
import re
from typing import Annotated, Any, Callable, Dict, List, Optional

from mcp import McpError
from mcp.types import ErrorData
from pydantic import Field

from src.tools.common_utils.path_utils import _resolve_session_id_from_mcp_context
from src.utils.error_codes import INTERNAL_ERROR, INVALID_REQUEST

from .action_format import format_action_string, tool_name_to_api_name
from .runtime import _cache_wb_modules, _tool_exec_lock, workbench_cwd
from .session_store import WorkbenchSessionStore

_TOOL_SPECS: Optional[List[Dict[str, Any]]] = None


def _extract_param_descriptions(docstring: str) -> Dict[str, str]:
    """Extract simple NumPy-style parameter descriptions from WorkBench docstrings."""
    descriptions: Dict[str, str] = {}
    if not docstring:
        return descriptions

    lines = inspect.cleandoc(docstring).splitlines()
    in_params = False
    current: Optional[str] = None
    collected: List[str] = []

    def flush() -> None:
        nonlocal current, collected
        if current:
            text = " ".join(part.strip() for part in collected if part.strip())
            descriptions[current] = text or current
        current = None
        collected = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if stripped == "Parameters":
            in_params = True
            continue
        if in_params and stripped in {"Returns", "Examples"}:
            flush()
            break
        if not in_params or not stripped or set(stripped) == {"-"}:
            continue

        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:?", stripped)
        is_param_line = bool(match) and not raw.startswith((" ", "\t"))
        if is_param_line:
            flush()
            current = match.group(1)
            after = stripped[match.end():].strip(" ,:-")
            if after:
                collected.append(after)
        elif current:
            collected.append(stripped)

    flush()
    return descriptions


def _load_tool_specs() -> List[Dict[str, Any]]:
    global _TOOL_SPECS
    if _TOOL_SPECS is not None:
        return _TOOL_SPECS

    with workbench_cwd():
        from src.tools.toolkits import all_tools  # noqa: WPS433
        _cache_wb_modules()

    specs: List[Dict[str, Any]] = []
    for lc_tool in all_tools:
        lc_name = lc_tool.name
        api_name = tool_name_to_api_name(lc_name)
        desc = (lc_tool.description or lc_tool.func.__doc__ or lc_name).strip()
        sig = inspect.signature(lc_tool.func)
        param_desc = _extract_param_descriptions(lc_tool.func.__doc__ or desc)
        properties: Dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            prop: Dict[str, Any] = {
                "type": "string",
                "description": param_desc.get(pname, pname),
            }
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            properties[pname] = prop
        specs.append({
            "lc_name": lc_name,
            "api_name": api_name,
            "func": lc_tool.func,
            "description": desc,
            "properties": properties,
        })
    _TOOL_SPECS = specs
    return specs


async def _require_session_id(ctx) -> str:
    session_id = await _resolve_session_id_from_mcp_context(ctx)
    if not session_id:
        raise McpError(
            ErrorData(code=INVALID_REQUEST, message="Missing session ID for WorkBench tool.")
        )
    return session_id


def _run_workbench_call(
    session_id: str,
    lc_name: str,
    func: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    action_str = format_action_string(lc_name, filtered)
    state = WorkbenchSessionStore.get_or_create(session_id)
    with _tool_exec_lock:
        state.apply_to_globals()
        state.function_calls.append(action_str)
        try:
            result = func(**filtered)
            if hasattr(result, "tolist"):
                result = result.tolist()
        except Exception as exc:
            state.capture_from_globals()
            return {
                "success": False,
                "data": None,
                "error": str(exc),
                "workbench_action": action_str,
            }
        state.capture_from_globals()
    return {
        "success": True,
        "data": {
            "result": result,
            "workbench_action": action_str,
        },
        "error": None,
    }


def _register_domain_tool(mcp, spec: Dict[str, Any]) -> None:
    """Create and register one WorkBench tool with a proper per-parameter schema."""
    api_name = spec["api_name"]
    lc_name = spec["lc_name"]
    func_ref = spec["func"]
    description = spec["description"]
    prop_names = list(spec["properties"].keys())

    param_defs = ", ".join(
        (
            f"{p}: Annotated[Optional[str], "
            f"Field(description={spec['properties'][p].get('description', p)!r})] = None"
        )
        for p in prop_names
    )
    kwargs_collect = ", ".join(f'"{p}": {p}' for p in prop_names)

    namespace: Dict[str, Any] = {
        "Optional": Optional,
        "Annotated": Annotated,
        "Field": Field,
        "_require_session_id": _require_session_id,
        "_run_workbench_call": _run_workbench_call,
        "asyncio": asyncio,
        "lc_name": lc_name,
        "func_ref": func_ref,
    }
    code = f'''
async def {api_name}({param_defs}, ctx=None) -> dict:
    session_id = await _require_session_id(ctx)
    kwargs = {{{kwargs_collect}}}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: _run_workbench_call(session_id, lc_name, func_ref, kwargs),
    )
'''
    exec(code, namespace)  # noqa: S102 — controlled template per WorkBench tool
    fn = namespace[api_name]
    fn.__doc__ = description
    mcp.tool(name=api_name, description=description, exclude_args=["ctx"])(fn)


def register_tools(mcp, session_middleware) -> None:
    """Register WorkBench domain tools and session helpers."""
    del session_middleware  # reserved for future session hooks

    @mcp.tool
    async def workbench_reset_state(ctx=None) -> dict:
        """
        Reset the WorkBench sandbox databases to baseline CSV state for this MCP session.
        Call at the start of each WorkBench task so prior tool calls do not leak state.
        """
        try:
            session_id = await _require_session_id(ctx)
            WorkbenchSessionStore.reset(session_id)
            return {
                "success": True,
                "data": {"message": "WorkBench sandbox reset.", "session_id": session_id},
                "error": None,
            }
        except McpError:
            raise
        except Exception as exc:
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"workbench_reset_state failed: {exc}")
            ) from exc

    @mcp.tool
    async def workbench_get_recorded_actions(ctx=None) -> dict:
        """
        Return all WorkBench action strings recorded for this session (official eval format).
        """
        try:
            session_id = await _require_session_id(ctx)
            actions = WorkbenchSessionStore.get_recorded_actions(session_id)
            return {
                "success": True,
                "data": {"function_calls": actions, "count": len(actions)},
                "error": None,
            }
        except McpError:
            raise
        except Exception as exc:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"workbench_get_recorded_actions failed: {exc}",
                )
            ) from exc

    for spec in _load_tool_specs():
        _register_domain_tool(mcp, spec)


def on_session_closed(session_id: str) -> None:
    """Hook: free WorkBench state when an MCP session is deleted."""
    WorkbenchSessionStore.remove(session_id)
