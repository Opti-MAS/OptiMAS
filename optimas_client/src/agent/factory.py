"""Agent factory: creates agent instances from config."""
import copy
import os
from pathlib import Path
from typing import Dict, Any, Optional, Type, Set

from .base_react_agent import BaseAgent
from ..workbench.query_builder import is_workbench_problem
from ..mcp.client import MCPClient
from ..config.loader import ConfigLoader
from ..messaging.emitter import MessageEmitter


def get_agent_class_by_name(
    agent_name: str,
    problem: Optional[Dict[str, Any]] = None,
) -> Type[BaseAgent]:
    """
    Resolve agent class by name.

    Only "optimizer" maps to OptimizerAgent; all other MAS agent names
    (dynamically created by the optimizer) use BaseAgent.
    WorkBench tasks use :class:`~..workbench.agent.WorkbenchAgent`.
    """
    if agent_name == "optimizer":
        from ..optimizer.optimizer_agent import OptimizerAgent
        return OptimizerAgent

    if problem is not None and is_workbench_problem(problem):
        from ..workbench.agent import WorkbenchAgent
        return WorkbenchAgent

    return BaseAgent


def create_agent(
    agent_name: str,
    mcp_client: MCPClient,
    config: Optional[Dict[str, Any]] = None,
    ancestor_agents: Optional[Set[str]] = None,
    problem: Optional[Dict[str, Any]] = None,
) -> BaseAgent:
    """
    Create an Agent instance from config.

    Args:
        agent_name: Agent name (must exist in config["agents"]).
        mcp_client: Shared MCP client instance.
        config: Full config dict. Loaded from default path if None.
        ancestor_agents: Names of agents in the current delegation chain.
            Passed through to the agent instance for runtime cycle prevention.
    """
    if config is None:
        config = ConfigLoader.load_config()

    agent_config = copy.deepcopy(ConfigLoader.get_agent_config(config, agent_name))
    agent_config["_config_dir"] = config.get("_config_dir", ".")
    agent_class = get_agent_class_by_name(agent_name, problem=problem)

    # Independent emitter per agent (deep copy avoids shared-state issues in concurrent runs)
    messaging_config = copy.deepcopy(ConfigLoader.get_messaging_config(config))
    workspace_base_dir = ConfigLoader.get_workspace_base_dir(config)
    handlers = messaging_config.get("handlers") or []
    if mcp_client.workspace_path is not None:
        path = str(Path(workspace_base_dir) / mcp_client.workspace_path / "working_trajectory")
        files = [h for h in handlers if isinstance(h, dict) and h.get("type") == "file"]
        if files:
            for handler in files:
                handler["base_path"] = path
        else:
            messaging_config.setdefault("handlers", []).append({"type": "file", "base_path": path})
    message_emitter = MessageEmitter.from_config(messaging_config)

    task_id = mcp_client.task_id
    invocation_id = os.urandom(8).hex()
    message_emitter.set_context(
        agent_name=agent_name,
        task_id=task_id,
        phase_count=0,
        invocation_id=invocation_id,
    )

    recording_config = ConfigLoader.get_api_recording_config(config)
    global_llm_settings = ConfigLoader.get_llm_settings(config)

    agent = agent_class(
        mcp_client=mcp_client,
        config=agent_config,
        message_emitter=message_emitter,
        recording_config=recording_config,
        global_llm_settings=global_llm_settings,
        global_config=config,
        ancestor_agents=ancestor_agents,
    )

    return agent
