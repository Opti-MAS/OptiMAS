"""MAS agent for WorkBench tasks (tools run on MCP server, not as client internals)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..agent.base_react_agent import BaseAgent

logger = logging.getLogger(__name__)


class WorkbenchAgent(BaseAgent):
    """WorkBench worker: domain tools are MCP tools on optimas_server; actions are recorded server-side."""

    async def execute_task(self, task: str, resume: bool = False, **kwargs) -> Dict[str, Any]:
        if not resume and not self._ancestor_agents:
            self.mcp_client.clear_workbench_actions()
            try:
                reset_res = await self.mcp_client.call_tool("workbench_reset_state", {})
                if not reset_res.get("success"):
                    return {"success": False, "data": None, "error": f"WorkBench reset failed: {reset_res.get('error')}"}
            except Exception as exc:
                return {"success": False, "data": None, "error": f"WorkBench reset failed: {exc}"}

        result = await super().execute_task(task, resume=resume, **kwargs)

        function_calls: List[str] = []
        try:
            rec = await self.mcp_client.call_tool("workbench_get_recorded_actions", {})
            if rec.get("success") and isinstance(rec.get("data"), dict):
                function_calls = list(rec["data"].get("function_calls") or [])
        except Exception as exc:
            logger.warning("workbench_get_recorded_actions failed: %s", exc)

        if not function_calls:
            function_calls = self.mcp_client.get_workbench_actions()

        if isinstance(result, dict):
            result = dict(result)
            result["function_calls"] = function_calls
        return result
