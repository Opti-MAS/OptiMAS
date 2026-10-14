"""Format agent tool invocations as WorkBench official action strings."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def tool_name_to_api_name(lc_tool_name: str) -> str:
    """``calendar.create_event`` -> ``calendar_create_event`` (valid Python identifier)."""
    return lc_tool_name.replace(".", "_")




def format_action_string(tool_fq_name: str, arguments: Dict[str, Any]) -> str:
    """Build ``module.tool.func(k="v", ...)`` matching WorkBench eval format."""
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


def parse_ground_truth_actions(answer: Any) -> List[str]:
    """Normalize stored gold actions to a list of action strings."""
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(a) for a in answer]
    if isinstance(answer, str):
        text = answer.strip()
        if not text:
            return []
        import ast

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except (SyntaxError, ValueError):
            pass
        return [text]
    return [str(answer)]
