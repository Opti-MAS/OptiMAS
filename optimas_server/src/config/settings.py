"""
Configuration management using Pydantic.

Supports:
- YAML configuration file (config.yaml or custom path)
- Default values
- Environment variables (.env file)
"""
import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file at module import time
load_dotenv(override=False)


PLACEHOLDER_SECRET_KEY = "optimas-server-secret-key-change-in-production"


class ServerSettings(BaseModel):
    """Server configuration"""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8188, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")


class AuthSettings(BaseModel):
    """Authentication configuration"""

    api_keys: List[str] = Field(
        default_factory=list, description="List of valid API keys"
    )



class SessionSettings(BaseModel):
    """Session management configuration"""

    timeout_seconds: int = Field(
        default=7200, description="Session timeout in seconds (2 hours)"
    )
    cleanup_interval: int = Field(
        default=300, description="Cleanup interval in seconds (5 minutes)"
    )
    max_sessions: int = Field(
        default=0, description="Maximum number of sessions (0 = unlimited)"
    )
    secret_key: str = Field(
        default=PLACEHOLDER_SECRET_KEY,
        description="Secret key for generating session_id (HMAC key)"
    )

    def is_placeholder_secret(self) -> bool:
        """True when ``secret_key`` is still a shipped placeholder."""
        return (
            self.secret_key == PLACEHOLDER_SECRET_KEY
            or self.secret_key.startswith("change-me")
        )


class LoggingSettings(BaseModel):
    """Logging configuration"""

    log_file: str = Field(
        default="logs/mcp_server.log", description="Log file path"
    )
    tool_calls_log_file: str = Field(
        default="logs/mcp_server_tools.jsonl",
        description="JSONL audit log for MCP tool calls (dashboard history + session stats)",
    )
    log_level: str = Field(
        default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)"
    )
    enable_file_logging: bool = Field(
        default=True, description="Enable file logging"
    )
    enable_console_logging: bool = Field(
        default=True, description="Enable console logging"
    )


class WorkspaceSettings(BaseModel):
    """Workspace configuration"""

    path: str = Field(
        default="./workspace",
        description="Root directory for session workspaces. Each session will have its own subdirectory named by session_id."
    )
    ignore_paths: List[str] = Field(
        default_factory=lambda: ["_internal", ".gitkeep", ".gitignore"],
        description="List of paths (files or directories) to ignore in list_file_tree. Paths are matched by name."
    )
    hide_own_working_trajectory_in_list: bool = Field(
        default=False,
        description=(
            "If true, list_file_tree hides this session's private log dirs only: "
            "<workspace>/working_trajectory, <workspace>/tool_call, and "
            "<workspace>/optimizer/working_trajectory, <workspace>/optimizer/tool_call — "
            "so agents are not prompted to read their own logs or tool audit files. "
            "Deeper paths such as <workspace>/<query_id>/working_trajectory or "
            "<workspace>/<query_id>/tool_call remain visible (e.g. optimizer at step_x "
            "listing per-query MAS workspaces). Override with env "
            "OPTIMAS_SERVER_HIDE_OWN_WORKING_TRAJECTORY_LIST."
        ),
    )


class DashboardSettings(BaseModel):
    """Dashboard availability. HTTP access requires an MCP API key.

    Browser clients use HTTP Basic authentication with the API key as password.
    Remote dashboard registration is an explicit opt-in.
    """

    enabled: bool = Field(
        default=True, description="Serve the web dashboard and its /api/* routes"
    )
    allow_remote: bool = Field(
        default=False,
        description=(
            "Permit the dashboard on a non-loopback bind. The dashboard is "
            "authenticated and can execute tools. Use a TLS reverse proxy and "
            "strong credentials for remote access."
        ),
    )


class Settings(BaseModel):
    """Application settings"""

    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path = "config.yaml") -> "Settings":
        """
        Load settings from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Settings instance

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file is invalid
        """
        yaml_path = Path(yaml_path).expanduser().resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}. "
                f"Please create a config.yaml file or specify a custom path."
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Parse nested settings with defaults
        server_data = yaml_data.get("server", {})
        auth_data = yaml_data.get("auth", {})
        session_data = yaml_data.get("session", {})
        logging_data = yaml_data.get("logging", {})
        workspace_data = dict(yaml_data.get("workspace") or {})
        dashboard_data = yaml_data.get("dashboard") or {}

        # Resolve storage once, before benchmark code changes the process cwd.
        for data, key, default in (
            (workspace_data, "path", "./workspace"),
            (logging_data, "log_file", "logs/mcp_server.log"),
            (logging_data, "tool_calls_log_file", "logs/mcp_server_tools.jsonl"),
        ):
            path = Path(data.get(key, default)).expanduser()
            data[key] = str((yaml_path.parent / path).resolve())

        if os.getenv("MCP_API_KEY"):
            auth_data["api_keys"] = [os.environ["MCP_API_KEY"]]
        if os.getenv("OPTIMAS_SESSION_SECRET"):
            session_data["secret_key"] = os.environ["OPTIMAS_SESSION_SECRET"]

        # Handle api_keys as string or list
        if "api_keys" in auth_data:
            api_keys = auth_data["api_keys"]
            if isinstance(api_keys, str):
                auth_data["api_keys"] = [
                    k.strip() for k in api_keys.split(",") if k.strip()
                ]

        # Optional: hide session-root working_trajectory + tool_call from list_file_tree (env wins over YAML)
        _env_hide_wt = os.getenv("OPTIMAS_SERVER_HIDE_OWN_WORKING_TRAJECTORY_LIST")
        if _env_hide_wt is not None:
            v = str(_env_hide_wt).strip().lower()
            workspace_data["hide_own_working_trajectory_in_list"] = v in (
                "1",
                "true",
                "yes",
                "on",
            )

        return cls(
            server=ServerSettings(**server_data),
            auth=AuthSettings(**auth_data),
            session=SessionSettings(**session_data),
            logging=LoggingSettings(**logging_data),
            workspace=WorkspaceSettings(**workspace_data),
            dashboard=DashboardSettings(**dashboard_data),
        )


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def load_settings(yaml_path: str | Path = "config.yaml") -> Settings:
    """
    Load and return settings instance (singleton pattern).

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings.load_from_yaml(yaml_path)
    return _settings


def get_settings() -> Settings:
    """
    Get current settings instance.

    Returns:
        Settings instance (raises error if not loaded)
    """
    if _settings is None:
        raise RuntimeError(
            "Settings not loaded. Call load_settings() first."
        )
    return _settings

