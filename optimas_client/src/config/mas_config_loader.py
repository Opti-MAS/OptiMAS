"""
MAS Configuration Loader.

Loads MAS configuration from folder structure and converts it to optimas_client format.
Supports directory configurations and single YAML files.
Provides DAG validation (entry agent, subagent references, cycle detection).
"""

import logging
import re
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)


def get_entry_agent_name_from_config(config: Dict[str, Any]) -> str:
    """
    Get the entry agent name from configuration.

    Args:
        config: MAS configuration dictionary

    Returns:
        The entry agent name

    Raises:
        ValueError: If no entry agent is found
    """
    for key, agent in config.get("agents", {}).items():
        if agent.get("is_entry_agent", False):
            return key
    raise ValueError("No entry agent found in configuration")


def get_agent_dependency_graph(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build adjacency list: agent_name -> list of subagent names it can delegate to."""
    agents = config.get("agents", {})
    graph: Dict[str, List[str]] = {}
    if isinstance(agents, dict):
        for name, agent_cfg in agents.items():
            subs = agent_cfg.get("subagent", [])
            graph[name] = list(subs) if isinstance(subs, list) else []
    elif isinstance(agents, list):
        for agent_cfg in agents:
            name = agent_cfg.get("agent_name", "")
            if not name:
                continue
            subs = agent_cfg.get("subagent", [])
            graph[name] = list(subs) if isinstance(subs, list) else []
    return graph


def _detect_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    """DFS-based cycle detection. Returns list of cycles found (each as a node path)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in graph}
    path: List[str] = []
    cycles: List[List[str]] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        path.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
            elif color[neighbor] == WHITE:
                dfs(neighbor)
        path.pop()
        color[node] = BLACK

    for node in graph:
        if color[node] == WHITE:
            dfs(node)
    return cycles


def validate_mas_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate MAS configuration for structural correctness.

    Checks:
      1. Exactly one entry agent exists.
      2. All subagent references point to defined agents.
      3. No cycles in the delegation graph.

    Returns:
        {"valid": bool, "errors": [str], "agent_graph": {name: [subagents]}}
    """
    errors: List[str] = []
    agents = config.get("agents", {})

    # Determine all defined agent names
    if isinstance(agents, dict):
        agent_names: Set[str] = set(agents.keys())
    elif isinstance(agents, list):
        agent_names = {a.get("agent_name", "") for a in agents} - {""}
    else:
        return {"valid": False, "errors": ["'agents' must be a dict or list"], "agent_graph": {}}

    if not agent_names:
        return {"valid": False, "errors": ["No agents defined in config"], "agent_graph": {}}

    blocks = agents.items() if isinstance(agents, dict) else ((a.get("agent_name", ""), a) for a in agents)
    for name, cfg in blocks:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,99}", name):
            errors.append("Agent names must be safe identifiers of 1-100 characters.")
        if name == "optimizer":
            errors.append("'optimizer' is reserved for the backward-pass agent.")
        if not isinstance(cfg, dict):
            errors.append(f"Agent '{name}' must be a mapping.")
            continue
        for field in ("subagent", "available_tools"):
            values = cfg.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                errors.append(f"Agent '{name}': {field} must be a list of names.")
        if not isinstance(cfg.get("is_entry_agent", False), bool):
            errors.append(f"Agent '{name}': is_entry_agent must be a boolean.")
    if errors:
        return {"valid": False, "errors": errors, "agent_graph": {}}

    # 1. Entry agent check
    entry_agents = []
    if isinstance(agents, dict):
        for name, cfg in agents.items():
            if cfg.get("is_entry_agent", False):
                entry_agents.append(name)
    elif isinstance(agents, list):
        for cfg in agents:
            if cfg.get("is_entry_agent", False):
                entry_agents.append(cfg.get("agent_name", "?"))
    if len(entry_agents) == 0:
        errors.append("No entry agent found (need exactly one agent with is_entry_agent: true)")
    elif len(entry_agents) > 1:
        errors.append(f"Multiple entry agents found: {entry_agents}. Need exactly one.")

    # 2. Build graph and check dangling references
    graph = get_agent_dependency_graph(config)
    for name, subs in graph.items():
        for sub in subs:
            if sub not in agent_names:
                errors.append(f"Agent '{name}' references undefined subagent '{sub}'")

    # 3. Cycle detection
    cycles = _detect_cycles(graph)
    for cycle in cycles:
        errors.append(f"Cycle detected in delegation graph: {' -> '.join(cycle)}")

    # 4. Dead agent detection (non-entry agents not referenced by any subagent list)
    all_referenced: Set[str] = set()
    for subs in graph.values():
        all_referenced.update(subs)
    for name in agent_names:
        if name not in entry_agents and name not in all_referenced:
            errors.append(
                f"Dead agent '{name}': not an entry agent and not in any agent's subagent list. "
                f"Wire it into a parent's subagent list or remove it."
            )

    # 5. assign_task_to_agent tool check: agents with subagents must have the tool
    for name in agent_names:
        agent_cfg = agents[name] if isinstance(agents, dict) else None
        if agent_cfg is None:
            continue
        subs = agent_cfg.get("subagent", [])
        tools = agent_cfg.get("available_tools", [])
        if subs and "assign_task_to_agent" not in tools:
            errors.append(
                f"Agent '{name}' has subagent(s) {subs} but 'assign_task_to_agent' "
                f"is not in its available_tools. Add it or delegation will silently fail."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "agent_graph": graph,
    }
