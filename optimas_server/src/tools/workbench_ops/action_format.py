"""WorkBench official action string formatting (matches paper eval harness)."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def tool_name_to_api_name(lc_tool_name: str) -> str:
    return lc_tool_name.replace(".", "_")


def format_action_string(tool_fq_name: str, arguments: Dict[str, Any]) -> str:
    args_out: List[str] = []
    for key, value in arguments.items():
        if value is None:
            continue
        if isinstance(value, str):
            args_out.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, bool):
            args_out.append(f"{key}={str(value)}")
        else:
            args_out.append(f"{key}={repr(value)}")
    return f"{tool_fq_name}.func(" + ", ".join(args_out) + ")"
