"""Register WorkBench LangChain tools as optimas_client internal MCP tools."""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .action_format import format_action_string, tool_name_to_api_name
from .env import ensure_workbench_on_path

if TYPE_CHECKING:
    from ..agent.base_react_agent import BaseAgent


@dataclass(frozen=True)
class WorkbenchToolSpec:
    lc_name: str
    api_name: str
    func: Callable[..., Any]
    description: str
    input_schema: Dict[str, Any]


_TOOL_SPECS: Optional[List[WorkbenchToolSpec]] = None


def _param_to_json_schema(param: inspect.Parameter) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "string"}
    if param.default is not inspect.Parameter.empty:
        schema["default"] = param.default
    return schema


def _build_input_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        prop = _param_to_json_schema(param)
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def load_tool_specs() -> List[WorkbenchToolSpec]:
    """Load all WorkBench toolkit tools (cached)."""
    global _TOOL_SPECS
    if _TOOL_SPECS is not None:
        return _TOOL_SPECS

    root = ensure_workbench_on_path()
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        from src.tools.toolkits import all_tools  # noqa: WPS433
    finally:
        os.chdir(old_cwd)

    specs: List[WorkbenchToolSpec] = []
    for lc_tool in all_tools:
        lc_name = lc_tool.name
        api_name = tool_name_to_api_name(lc_name)
        desc = (lc_tool.description or lc_tool.func.__doc__ or lc_name).strip()
        specs.append(
            WorkbenchToolSpec(
                lc_name=lc_name,
                api_name=api_name,
                func=lc_tool.func,
                description=desc,
                input_schema=_build_input_schema(lc_tool.func),
            )
        )
    _TOOL_SPECS = specs
    return specs




def register_workbench_tools(agent: "BaseAgent", function_calls: List[str]) -> None:
    """Attach one ``@mcp_tool`` method per WorkBench tool onto *agent*."""
    from ..agent.utils.tool_decorator import mcp_tool

    for spec in load_tool_specs():
        if hasattr(agent, spec.api_name):
            continue

        def _make_handler(
            _spec: WorkbenchToolSpec,
            _calls: List[str],
        ):
            def handler(agent_self, **kwargs):
                filtered = {k: v for k, v in kwargs.items() if v is not None}
                action_str = format_action_string(_spec.lc_name, filtered)
                _calls.append(action_str)
                try:
                    result = _spec.func(**filtered)
                except Exception as exc:
                    return {"success": False, "data": None, "error": str(exc)}
                return {"success": True, "data": result, "error": None}

            handler.__name__ = _spec.api_name
            handler.__doc__ = _spec.description
            return handler

        handler = _make_handler(spec, function_calls)
        decorated = mcp_tool(input_schema=spec.input_schema)(handler)
        bound = decorated.__get__(agent, type(agent))
        setattr(agent, spec.api_name, bound)
