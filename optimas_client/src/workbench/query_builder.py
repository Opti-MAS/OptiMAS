"""WorkBench task prompt helpers."""
from __future__ import annotations

from typing import Any, Dict

from .env import get_hardcoded_current_time


def is_workbench_problem(problem: Dict[str, Any]) -> bool:
    qt = str(problem.get("query_type", "")).lower()
    bt = str(problem.get("benchmark_type", "")).lower()
    return qt == "workbench" or bt == "workbench"


def build_workbench_query(problem: Dict[str, Any]) -> str:
    """Prefix natural-language task with WorkBench date/time context (official inference)."""
    base = problem.get("query", problem.get("question", ""))
    current = get_hardcoded_current_time()
    prefix = (
        f"Today's date is {current.strftime('%A')}, {current.date()} "
        f"and the current time is {current.time()}. "
        f"Remember the current date and time when answering queries. "
        f"Meetings must not start before 9am or end after 6pm.\n\n"
    )
    return prefix + base
