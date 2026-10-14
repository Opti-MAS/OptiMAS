"""System operations tools registration."""

from src.tools.common_utils.path_utils import set_session_middleware
from .system_tools import _register_system_tools


def register_tools(mcp, session_middleware):
    """
    Register system-level tools to FastMCP instance.

    Args:
        mcp: FastMCP instance
        session_middleware: SessionMiddleware instance (for workspace scoping)
    """
    set_session_middleware(session_middleware)
    _register_system_tools(mcp)
