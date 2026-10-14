"""
Resources module - registers all resources to FastMCP instance.

Usage:
    from src.resources import register_all_resources
    
    mcp = FastMCP("OptiMAS Server")
    register_all_resources(mcp)
"""
from fastmcp import FastMCP


def register_all_resources(mcp: FastMCP):
    """
    Register all resources to FastMCP instance.

    Args:
        mcp: FastMCP instance
    """
    from . import resource_handlers

    resource_handlers.register_resources(mcp)
