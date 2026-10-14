"""Configuration loader with YAML and .env support."""
import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from dotenv import load_dotenv

# Load .env file at module import time
load_dotenv(override=False)

from ..utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_LLM_SETTINGS = {
    "read_timeout": 60.0
}
DEFAULT_API_USAGE_RECORDING_CONFIG = {
    "enabled": True,
    "storage_path": "API_USAGE_STORAGE"
}
DEFAULT_MESSAGING_CONFIG = {
    "handlers": [
        {
            "type": "console",
            "use_colors": True
        }
    ]
}
DEFAULT_MCP_CONFIG = {
    "server_url": "http://127.0.0.1:8188/mcp",
    "api_key": "optimas-local-dev-key"
}
DEFAULT_WORKSPACE_BASE_DIR = "./workspace"
# Skills are stored inside each agent's config directory (e.g.
# mas_workbench/skills/worker/), not in a global optimas_client/skills/ folder.
# DEFAULT_SKILL_DIR is kept as an empty string so get_skill_dir returns a
# falsy value when no top-level skill_dir is configured.
DEFAULT_SKILL_DIR = ""


def _deep_merge_dict_defaults_first(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """*override* wins on every key; *defaults* only supply missing branches (recursive for dicts)."""
    out = copy.deepcopy(defaults)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict_defaults_first(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigLoader:
    """Configuration loader with YAML and .env support."""

    # Optional: same layout as ``agents`` in ``config.yaml``; merged per agent id.
    # ``config.yaml`` wins on overlapping keys (recursive). Filename = per-agent LLM-related slice.
    AGENTS_LLM_CONFIG_YAML = "agents_llm_config.yaml"

    @staticmethod
    def _resolve_config_path(config_path: str) -> Path:
        """Resolve a config_path to an actual YAML file, searching common filenames if it's a directory."""
        p = Path(config_path)
        if p.is_file():
            return p
        for filename in ["config.yaml", "configs.yaml", "mas_config.yaml"]:
            candidate = p / filename
            if candidate.exists():
                return candidate
        return p  # caller will raise if not found

    @staticmethod
    def _load_yaml(config_path: Path) -> Dict[str, Any]:
        """Load and parse a single YAML file. Raises on any error."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ConfigError(f"YAML root must be a mapping, got {type(config).__name__}")
        config["_config_dir"] = str(config_path.parent.absolute())
        return config

    @staticmethod
    def load_config(
        config_path: str = "optimas_client/config/global_config.yaml",
    ) -> Dict[str, Any]:
        """
        Load configuration.
        
        Args:
            config_path: Path to YAML config file
        
        Raises:
            ConfigError: If configuration loading fails
        """
        resolved = ConfigLoader._resolve_config_path(config_path)

        if not resolved.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        try:
            config = ConfigLoader._load_yaml(resolved)
        except Exception as e:
            raise ConfigError(f"Failed to load YAML config from {resolved}: {e}") from e

        config = ConfigLoader._merge_env_vars(config)
        ConfigLoader._merge_agents_llm_config_file(config)
        return config

    @staticmethod
    def load_config_safe(
        config_path: str,
        fallback_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration with automatic fallback on YAML parse failure.

        Used in training loops to load optimizer-generated MAS configs: even if
        the optimizer produces broken YAML, training won't crash but falls back
        to the last valid config.

        Raises ConfigError if fallback also fails or is not provided.
        """
        resolved = ConfigLoader._resolve_config_path(config_path)
        try:
            config = ConfigLoader._load_yaml(resolved)
            config = ConfigLoader._merge_env_vars(config)
            ConfigLoader._merge_agents_llm_config_file(config)
            return config
        except Exception as primary_err:
            logger.error("Failed to load config from %s: %s", resolved, primary_err)

            if not fallback_config_path:
                raise ConfigError(
                    f"Failed to load config from {resolved} and no fallback provided: {primary_err}"
                ) from primary_err

            fallback_resolved = ConfigLoader._resolve_config_path(fallback_config_path)
            logger.warning("Attempting fallback config from %s", fallback_resolved)
            try:
                fallback_config = ConfigLoader._load_yaml(fallback_resolved)
                fallback_config = ConfigLoader._merge_env_vars(fallback_config)
                ConfigLoader._merge_agents_llm_config_file(fallback_config)

                # Overwrite the broken file with the valid fallback so subsequent
                # reads within this step also succeed.
                if resolved.exists():
                    broken_backup = resolved.with_suffix(".yaml.broken")
                    shutil.copy2(resolved, broken_backup)
                    logger.warning("Broken config backed up to %s", broken_backup)
                shutil.copy2(fallback_resolved, resolved)
                logger.warning("Restored config from fallback: %s -> %s", fallback_resolved, resolved)

                return fallback_config
            except Exception as fallback_err:
                raise ConfigError(
                    f"Primary config failed ({primary_err}) and fallback also failed ({fallback_err})"
                ) from fallback_err
    
    @staticmethod
    def _merge_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration with environment variables.
        
        Args:
            config: Original config dict
        
        Returns:
            Merged config dict
        """
        # MCP config
        if "mcp" not in config:
            config["mcp"] = {}
        
        if os.getenv("MCP_API_KEY"):
            config["mcp"]["api_key"] = os.getenv("MCP_API_KEY")
        
        # API key and base_url are now read directly from env vars in factory.py
        
        return config

    @staticmethod
    def _merge_agents_llm_config_file(config: Dict[str, Any]) -> None:
        """If ``agents_llm_config.yaml`` exists beside the main YAML, merge each ``agents.<id>`` into *config*.

        Same ``agents:`` shape as ``config.yaml``. Per agent id, values from ``agents_llm_config.yaml``
        are applied first, then the agent block from the loaded main file overwrites on conflicts
        (recursive for nested mappings).
        """
        cfg_dir = config.get("_config_dir")
        if not cfg_dir:
            return
        sidecar = Path(cfg_dir) / ConfigLoader.AGENTS_LLM_CONFIG_YAML
        if not sidecar.is_file():
            return
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Skipped %s: %s", sidecar, e)
            return
        if not isinstance(raw, dict):
            return
        llm_agents = raw.get("agents")
        if not isinstance(llm_agents, dict):
            return
        main_agents = config.get("agents")
        if not isinstance(main_agents, dict):
            return
        for agent_id, main_agent in main_agents.items():
            if not isinstance(main_agent, dict):
                continue
            llm_agent = llm_agents.get(agent_id)
            if isinstance(llm_agent, dict) and llm_agent:
                main_agents[agent_id] = _deep_merge_dict_defaults_first(llm_agent, main_agent)

    @staticmethod
    def get_agent_llm_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """
        Get the LLM configuration for a specific agent.
        
        Args:
            config: Full config dict
            agent_name: Agent name
        
        Returns:
            LLM config dict
        
        Raises:
            ConfigError: If agent or LLM config not found
        """
        agent_config = ConfigLoader.get_agent_config(config, agent_name)
        
        llm_config = agent_config.get("llm_config", {})
        if not llm_config:
            raise ConfigError(f"LLM config not found for agent '{agent_name}'")
        
        return llm_config
    
    @staticmethod
    def get_mcp_config(config: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get MCP configuration.
        
        Args:
            config: Full config dict
            task_id: Task ID (optional; uses default if not provided)
        
        Returns:
            MCP config dict
        
        Raises:
            ConfigError: If MCP config is missing or lacks required fields
        """
        mcp_config = config.get("mcp", DEFAULT_MCP_CONFIG)
        if not mcp_config:
            raise ConfigError("MCP config not found in configuration")
        
        # Check required fields
        if "server_url" not in mcp_config:
            raise ConfigError("MCP config missing required field: server_url")
        
        if "api_key" not in mcp_config:
            raise ConfigError("MCP config missing required field: api_key (should be set in .env as MCP_API_KEY)")
        
        # Set task_id (use provided value, otherwise use default)
        mcp_config["task_id"] = task_id or mcp_config.get("task_id", "default-task")
        
        return mcp_config
    
    @staticmethod
    def get_agent_config(config: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent.
        
        Args:
            config: Full config dict
            agent_name: Agent name
        
        Returns:
            Agent config dict (contains max_iterations, available_tools, llm_invisible_tools, etc.)
        
        Raises:
            ConfigError: If agent not found
        """
        agents = config.get("agents")
        if isinstance(agents, dict):
            if agents.get(agent_name, None):
                return agents.get(agent_name)
        elif isinstance(agents, list):
            for agent in agents:
                if agent.get("agent_name") == agent_name:
                    return agent
        
        raise ConfigError(f"Agent '{agent_name}' not found in configuration")
    
    @staticmethod
    def list_agents(config: Dict[str, Any]) -> List[str]:
        """
        List all configured agent names.
        
        Args:
            config: Full config dict
        
        Returns:
            List of agent names
        """
        agents = config.get("agents", [])
        if isinstance(agents, dict):
            return list(agents.keys())
        return [agent.get("agent_name") for agent in agents if agent.get("agent_name")]
    
    @staticmethod
    def get_messaging_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get messaging configuration.
        
        Args:
            config: Full config dict
        
        Returns:
            Messaging config dict (returns default if not present)
        """
        return config.get("messaging", DEFAULT_MESSAGING_CONFIG)
    
    @staticmethod
    def get_api_recording_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get API usage recording configuration.
        
        Args:
            config: Full config dict
        
        Returns:
            API usage recording config dict with 'enabled' and 'storage_path'
        """
        recording_config = config.get("api_usage_recording", DEFAULT_API_USAGE_RECORDING_CONFIG)
        default_config = {
            "enabled": True,
            "storage_path": None  # Falls back to logs/api_usage under project root
        }
        # Merge config
        return {**default_config, **recording_config}
    
    @staticmethod
    def get_llm_settings(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM global settings.
        
        Args:
            config: Full config dict
        
        Returns:
            LLM global settings dict (e.g. 'timeout')
        """
        llm_settings = config.get("llm_settings", DEFAULT_LLM_SETTINGS)
        # Return raw config (BaseLLM handles merge logic)
        return llm_settings

    @staticmethod
    def get_workspace_base_dir(config: Dict[str, Any]) -> str:
        """
        Get workspace base directory.
        
        Args:
            config: Full config dict
        
        Returns:
            Workspace base directory path
        """
        return config.get("workspace_base_dir", DEFAULT_WORKSPACE_BASE_DIR)

    @staticmethod
    def get_skill_dir(config: Dict[str, Any]) -> str:
        """
        Get skill directory.
        
        Args:
            config: Full config dict
        
        Returns:
            Skill directory path
        """
        return config.get("skill_dir", DEFAULT_SKILL_DIR)