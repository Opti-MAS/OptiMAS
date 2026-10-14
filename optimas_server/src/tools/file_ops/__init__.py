"""File operations tools"""

from src.tools.common_utils.path_utils import set_session_middleware


def register_tools(mcp, session_middleware):
    """
    Register all file operation tools to FastMCP instance.
    
    Args:
        mcp: FastMCP instance
        session_middleware: SessionMiddleware instance
    """
    # Set session middleware for utility functions
    set_session_middleware(session_middleware)
    
    # Import and register tools from each module (lazy import to avoid circular dependencies)
    from .read_write_tools import _register_read_write_tools
    from .edit_tools import _register_edit_tools
    from .lookup_tools import _register_search_tools

    # Register tools from each module
    _register_read_write_tools(mcp)
    _register_edit_tools(mcp)
    _register_search_tools(mcp)


__all__ = ["register_tools"]
