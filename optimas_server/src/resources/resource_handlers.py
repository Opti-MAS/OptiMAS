"""Public MCP resources containing non-sensitive service information."""
import platform
from datetime import datetime
from typing import Any, Dict


def register_resources(mcp):
    """Expose service metadata; credentials and session logs stay private."""

    @mcp.resource("file:///config/server_config")
    def get_server_config() -> Dict[str, Any]:
        """Return transport settings without credentials or storage paths."""
        from src.config import get_settings

        settings = get_settings()
        return {
            "server": {"port": settings.server.port, "debug": settings.server.debug},
            "session": {"timeout_seconds": settings.session.timeout_seconds},
        }

    @mcp.resource("file:///system/info")
    def get_system_info_resource() -> Dict[str, Any]:
        """Return runtime version information without host-specific paths."""
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "current_time": datetime.now().isoformat(),
        }
