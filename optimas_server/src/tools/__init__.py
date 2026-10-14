"""
Tools module - registers all tools to FastMCP instance.

Usage:
    from src.tools import register_all_tools
    from src.middleware import SessionMiddleware
    
    mcp = FastMCP("OptiMAS Server")
    session_middleware = SessionMiddleware(...)
    register_all_tools(mcp, session_middleware)
"""
from fastmcp import FastMCP


def register_all_tools(mcp: FastMCP, session_middleware):
    """
    Register all tools to FastMCP instance.

    Args:
        mcp: FastMCP instance
        session_middleware: SessionMiddleware instance
    """
    # Import and register tools from each module
    from .file_ops import register_tools as register_file_tools
    from .system_ops import register_tools as register_system_tools
    from .session_ops import session_tools
    from .workbench_ops import register_tools as register_workbench_tools

    # Register tools
    register_file_tools(mcp, session_middleware)
    register_system_tools(mcp, session_middleware)
    session_tools.register_tools(mcp, session_middleware)
    register_workbench_tools(mcp, session_middleware)
