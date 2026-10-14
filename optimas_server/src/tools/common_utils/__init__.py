"""
Common utilities for tools.

Provides shared functions for path handling and workspace management.
"""
from .path_utils import (
    set_session_middleware,
    get_session_workspace,
    resolve_workspace_path,
    get_client_path,
)

__all__ = [
    "set_session_middleware",
    "get_session_workspace",
    "resolve_workspace_path",
    "get_client_path",
]

