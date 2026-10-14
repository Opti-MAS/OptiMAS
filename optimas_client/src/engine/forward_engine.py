"""
MAS Forward Engine

Handles concurrent execution of the Multi-Agent System on a batch of problems.
This engine is purely responsible for executing the MAS and collecting results,
without any evaluation logic.

Concurrency model:
  Uses asyncio.Semaphore + asyncio.gather for bounded concurrent execution.
  Each problem gets its own MCPClient session and event loop context.
  No ProcessPoolExecutor/ThreadPoolExecutor — MAS execution is I/O-bound (LLM API
  calls, MCP tool calls), so native asyncio concurrency is both simpler and faster.

Safety:
  - Per-problem timeout prevents a single stuck agent from blocking the batch.
  - Each problem is wrapped in a top-level try/except so failures are isolated.
  - Results preserve batch order regardless of completion order or failures.
"""

import asyncio
import copy
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..agent.factory import create_agent
from ..mcp.client import MCPClient
from ..config.mas_config_loader import get_entry_agent_name_from_config, validate_mas_config
from ..workbench.query_builder import is_workbench_problem, build_workbench_query

logger = logging.getLogger(__name__)

# Runtime defaults can be shared; agent definitions and prompt roots stay local to the MAS.
_GLOBAL_RUNTIME_KEYS = frozenset({
    "mcp", "messaging", "api_usage_recording", "llm_settings", "workspace_base_dir",
    "max_llm_concurrency", "per_problem_timeout",
})

_FAIL_RESULT: Dict[str, Any] = {"success": False, "data": None, "error": ""}


def _make_error(error: str) -> Dict[str, Any]:
    return {**_FAIL_RESULT, "error": error}


async def run_mas_on_problem(
    problem_id: str,
    problem: Dict[str, Any],
    mas_config: Dict[str, Any],
    mcp_server_url: str,
    mcp_api_key: str,
    workspace_base_dir: str,
    workspace_root: Optional[str] = None,
    timeout: Optional[float] = None,
    global_config: Optional[Dict[str, Any]] = None,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Run MAS on a single problem with full isolation.

    Returns:
        {"success": bool, "data": Any, "error": str|None, "wall_time_sec": float}

        Complete agent trajectories (including all phases and sub-agents) are
        persisted to ``working_trajectory/*.jsonl`` via ``MessageEmitter`` during
        execution — NOT carried in this return dict.  The in-memory
        ``conversation_history`` only holds the last phase and the entry agent,
        so it is unsuitable for statistics.

    Args:
        timeout: Per-problem timeout in seconds.  None = no timeout.
    """
    workbench_mode = is_workbench_problem(problem)

    async def _inner() -> Dict[str, Any]:
        config = copy.deepcopy(mas_config)
        if global_config is not None:
            for key, value in global_config.items():
                if key not in _GLOBAL_RUNTIME_KEYS:
                    continue
                config[key] = value
        if workspace_root is not None:
            config["workspace_base_dir"] = str(Path(workspace_root).resolve())

        ws_subdir = workspace_base_dir

        async with MCPClient(
            server_url=mcp_server_url,
            api_key=mcp_api_key,
            task_id=str(problem_id) if not test_mode else f"test-{problem_id}",
            workspace_subdir=ws_subdir,
        ) as mcp_client:

            entry_agent_name = get_entry_agent_name_from_config(config)
            entry_agent = create_agent(
                agent_name=entry_agent_name,
                mcp_client=mcp_client,
                config=config,
                problem=problem,
            )

            if workbench_mode:
                query = build_workbench_query(problem)
            else:
                query = problem.get("query", problem.get("question", ""))

            t_wall0 = time.perf_counter()
            result = await entry_agent.execute_task(task=query)
            wall_time_sec = time.perf_counter() - t_wall0

            if result is None:
                return {
                    **_FAIL_RESULT,
                    "error": "execute_task returned None",
                    "wall_time_sec": wall_time_sec,
                }
            if not isinstance(result, dict):
                return {
                    **_FAIL_RESULT,
                    "error": f"execute_task returned non-dict: {type(result).__name__}",
                    "wall_time_sec": wall_time_sec,
                }

            out: Dict[str, Any] = {
                "success": result.get("success", False),
                "data": result.get("data"),
                "error": result.get("error"),
                "wall_time_sec": wall_time_sec,
            }
            if workbench_mode and result.get("function_calls") is not None:
                out["function_calls"] = result["function_calls"]
            return out

    try:
        if timeout is not None and timeout > 0:
            ret = await asyncio.wait_for(_inner(), timeout=timeout)
        else:
            ret = await _inner()
    except asyncio.TimeoutError:
        logger.error("Problem %s timed out after %.0fs", problem_id, timeout)
        ret = _make_error(f"Timed out after {timeout}s")
    except Exception as e:
        logger.error("Problem %s failed: %s", problem_id, e, exc_info=True)
        ret = _make_error(str(e))

    return ret


class MASForwardEngine:
    def __init__(
        self,
        mcp_server_url: str,
        mcp_api_key: str,
        workspace_root: str,
        max_concurrent: int = 8,
        per_problem_timeout: Optional[float] = None,
        # Deprecated — kept for config compatibility, ignored at runtime.
        use_process_pool: bool = True,
        test_mode: bool = False,
    ):
        self.mcp_server_url = mcp_server_url
        self.mcp_api_key = mcp_api_key
        self.workspace_root = workspace_root
        self.max_concurrent = max_concurrent
        self.per_problem_timeout = per_problem_timeout
        self.test_mode = test_mode
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive.")

    async def forward(
        self,
        batch: List[Dict[str, Any]],
        mas_config: Dict[str, Any],
        workspace_subdir: str = "",
        global_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute forward pass on a batch of problems.

        Returns:
            results_list — same order as *batch*.  Each item is
            ``{"success", "data", "error", "wall_time_sec"}``.
        """
        if not batch:
            return []
        validation = validate_mas_config(mas_config)
        if not validation["valid"]:
            raise ValueError("Invalid MAS configuration: " + "; ".join(validation["errors"]))
        ids = [str(problem.get("query_id", "")) for problem in batch]
        if len(set(ids)) != len(ids):
            raise ValueError("Forward batch contains duplicate query IDs.")

        items: List[Tuple[str, Dict[str, Any]]] = []
        for problem in batch:
            problem_id = problem.get("query_id")
            if not problem_id:
                logger.warning("Problem missing query_id, skipping.")
                continue
            items.append((problem_id, problem))

        logger.info(
            "Running forward pass on %d problems (max_concurrent=%d, timeout=%s).",
            len(items), self.max_concurrent, self.per_problem_timeout,
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _guarded(pid: str, problem: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                result = await run_mas_on_problem(
                    problem_id=pid,
                    problem=problem,
                    mas_config=mas_config,
                    mcp_server_url=self.mcp_server_url,
                    mcp_api_key=self.mcp_api_key,
                    workspace_base_dir=workspace_subdir,
                    workspace_root=self.workspace_root,
                    timeout=self.per_problem_timeout,
                    global_config=global_config,
                    test_mode=self.test_mode,
                )
                return (pid, result)

        raw_results = await asyncio.gather(
            *[_guarded(pid, problem) for pid, problem in items],
            return_exceptions=True,
        )

        # Build lookup map — exceptions become error results
        results_map: Dict[str, Dict[str, Any]] = {}
        for i, raw in enumerate(raw_results):
            pid = items[i][0]
            if isinstance(raw, Exception):
                logger.error("Problem %s raised unhandled exception: %s", pid, raw)
                results_map[pid] = _make_error(f"Unhandled exception: {raw}")
            elif isinstance(raw, tuple) and len(raw) == 2:
                results_map[raw[0]] = raw[1]
            else:
                logger.error("Unexpected result shape for %s: %s", pid, type(raw))
                results_map[pid] = _make_error(f"Unexpected result type: {type(raw).__name__}")

        # Reconstruct results in batch order
        results_list: List[Dict[str, Any]] = []
        for problem in batch:
            pid = problem.get("query_id")
            res = results_map.get(pid, _make_error("Execution missing (no query_id)"))
            results_list.append(res)

        succeeded = sum(1 for r in results_list if r.get("success"))
        logger.info("Forward pass complete: %d/%d succeeded.", succeeded, len(results_list))

        return results_list
