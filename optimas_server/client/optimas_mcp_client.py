"""
OptiMAS Client - High-level client wrapper for OptiMAS Server.

Provides simplified API for connecting, handshaking, and calling tools.
"""
import copy
from typing import Optional, Any, Dict

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp import McpError
from mcp.types import ErrorData


class OptiMASClient:
    """
    High-level client for OptiMAS Server.
    
    Features:
    - Automatic connection management via async context manager
    - Automatic handshake on connection
    - Simplified tool calling with automatic session_id handling
    - Tool listing support
    - Tool schema retrieval
    
    Example:
        async with OptiMASClient(
            server_url="http://127.0.0.1:8188/mcp",
            api_key="your-api-key",
            task_id="task-001"
        ) as client:
            tools = await client.list_tools()
            schema = await client.get_tool_schema("read_file")
            result = await client.call_tool("echo", {"message": "Hello"})
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: str,
        task_id: str,
    ):
        """
        Initialize OptiMAS Client.
        
        Args:
            server_url: MCP server URL (e.g., "http://127.0.0.1:8188/mcp")
            api_key: API key for authentication
            task_id: Task identifier for session creation
        """
        self.server_url = server_url
        self.api_key = api_key
        self.task_id = task_id
        
        self._client: Optional[Client] = None
        self._transport: Optional[StreamableHttpTransport] = None
        self._session_id: Optional[str] = None
        self._connected = False
    
    async def __aenter__(self):
        """Enter async context manager - establish connection and handshake."""
        # Create transport
        self._transport = StreamableHttpTransport(
            url=self.server_url,
            headers={"X-MCP-Key": self.api_key},
        )
        
        # Create client
        self._client = Client(transport=self._transport)
        
        # Establish connection
        await self._client.__aenter__()
        self._connected = True
        
        # Perform handshake
        await self._handshake()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager - close connection."""
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._connected = False
            self._session_id = None
    
    async def _handshake(self) -> None:
        """
        Perform handshake to establish session.
        
        Raises:
            McpError: If handshake fails
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async with OptiMASClient(...)")
        
        result = await self._client.call_tool("handshake", {"task_id": self.task_id})
        
        if result.is_error:
            error_msg = "Handshake failed"
            if result.content and len(result.content) > 0:
                error_msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            raise McpError(
                ErrorData(code=-32603, message=error_msg)  # Internal error
            )
        
        # Extract session_id from result.data
        if hasattr(result, 'data') and isinstance(result.data, dict):
            self._session_id = result.data.get('session_id')
            if not self._session_id:
                raise McpError(
                    ErrorData(code=-32603, message="Handshake succeeded but session_id not found in response")
                )
        else:
            raise McpError(
                ErrorData(code=-32603, message="Handshake succeeded but invalid response format")
            )
    
    @property
    def session_id(self) -> Optional[str]:
        """
        Get current session ID.
        
        Returns:
            Session ID if connected and handshake completed, None otherwise
        """
        return self._session_id
    
    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """
        Call a tool with automatic session_id handling.
        
        Args:
            name: Tool name
            arguments: Tool arguments dictionary
            
        Returns:
            Tool call result (same format as FastMCP Client.call_tool)
            
        Raises:
            RuntimeError: If client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with OptiMASClient(...)")
        
        if not self._session_id:
            raise RuntimeError("Session not established. Handshake may have failed.")
        
        # Call tool with session_id in meta
        return await self._client.call_tool(
            name,
            arguments,
            meta={"session_id": self._session_id}
        )
    
    async def list_tools(self) -> list:
        """
        List all available tools.
        
        Returns:
            List of tool objects (same format as FastMCP Client.list_tools)
            
        Raises:
            RuntimeError: If client is not connected
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with OptiMASClient(...)")
        
        return await self._client.list_tools()

    @staticmethod
    def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize JSON Schema to be LLM-compatible by removing meta fields.
        
        Args:
            schema: JSON Schema dictionary to sanitize
            
        Returns:
            Sanitized schema dictionary with meta fields removed
        """
        if not isinstance(schema, dict):
            return schema
        
        new_schema = copy.deepcopy(schema)
        
        # Remove meta field if present
        if "meta" in new_schema:
            del new_schema["meta"]
        
        return new_schema

    async def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get schemas for all available tools with LLM-compatible sanitization.
        
        Returns a dictionary mapping tool names to their schema information.
        Schemas are sanitized to remove meta fields (LLM-compatible).
        
        Returns:
            A dict mapping tool name -> tool schema dictionary with:
            - name: Tool name
            - description: Tool description
            - inputSchema: Input parameters schema (sanitized, meta fields removed)
            
        Raises:
            RuntimeError: If client is not connected
            
        Example:
            schemas = await client.get_tool_schemas()
            read_file_schema = schemas["read_file"]
            print(read_file_schema["inputSchema"])
        """
        if not self._connected or self._client is None:
            raise RuntimeError("Client not connected. Use async with OptiMASClient(...)")

        tools = await self._client.list_tools()
        schemas: Dict[str, Dict[str, Any]] = {}

        for tool in tools:
            # Tool objects from FastMCP already contain schema information
            # See: https://gofastmcp.com/clients/tools
            schema: Dict[str, Any] = {
                "name": tool.name,
                "description": tool.description or "",
            }
            
            # Include and sanitize inputSchema if available
            if tool.inputSchema:
                schema["inputSchema"] = self._sanitize_schema(tool.inputSchema)
            
            schemas[schema["name"]] = schema

        return schemas

    async def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific tool.
        
        Args:
            tool_name: Name of the tool to get schema for
            
        Returns:
            Tool schema dictionary containing name, description, inputSchema, etc.
            Returns None if tool not found.
            
        Raises:
            RuntimeError: If client is not connected
            
        Example:
            schema = await client.get_tool_schema("read_file")
            if schema:
                print(schema["inputSchema"])
        """
        schemas = await self.get_tool_schemas()
        return schemas.get(tool_name)


