"""
Authentication middleware for API Key validation.
"""
from typing import List, Set

from fastmcp.server.middleware import Middleware, MiddlewareContext
from mcp import McpError
from mcp.types import ErrorData

from src.utils.error_codes import AUTH_MISSING, AUTH_INVALID


class AuthMiddleware(Middleware):
    """
    Middleware for API Key authentication.

    Uses X-MCP-Key HTTP header for authentication.
    """

    def __init__(
        self,
        api_keys: List[str],
    ):
        """
        Initialize authentication middleware.

        Args:
            api_keys: List of valid API keys
        """
        self.api_keys: Set[str] = set(api_keys) if api_keys else set()

    async def on_request(
        self, context: MiddlewareContext, call_next
    ) -> any:
        """
        Validate API Key on all requests (except initialize).
        
        Note: initialize requests skip authentication as headers may not be available
        at that stage. Authentication is enforced on all subsequent requests.
        """
        # Skip authentication for initialize requests
        method = None
        if hasattr(context, 'message') and hasattr(context.message, 'name'):
            method = context.message.name
        elif hasattr(context, 'method'):
            method = context.method
        
        if method == "initialize":
            return await call_next(context)
        
        # For all other requests, validate API Key
        try:
            from fastmcp.server.dependencies import get_http_headers
            headers = get_http_headers()
        except Exception:
            headers = {}

        api_key = self._extract_api_key(headers)
        if not api_key:
            raise McpError(
                ErrorData(code=AUTH_MISSING, message="Authentication failed: Missing API Key")
            )
        if api_key not in self.api_keys:
            raise McpError(
                ErrorData(code=AUTH_INVALID, message="Authentication failed: Invalid API Key")
            )

        return await call_next(context)

    def _extract_api_key(self, headers: dict) -> str | None:
        """Extract API Key from X-MCP-Key HTTP header (case-insensitive)."""
        headers_lower = {k.lower(): v for k, v in headers.items()}
        
        # Get X-MCP-Key header
        api_key = headers_lower.get("x-mcp-key")
        if api_key:
            return str(api_key).strip()

        return None

