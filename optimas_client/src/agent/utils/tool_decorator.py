"""Tool decorator for auto-registering internal agent tools."""
import inspect
from functools import wraps
from typing import Callable, Dict, Any, List, Optional


def mcp_tool(
    func: Optional[Callable] = None,
    *,
    input_schema: Dict[str, Any]
) -> Callable:
    """Decorator that marks an agent method as an MCP tool.

    Args:
        func: The decorated function.
        input_schema: Dict mapping parameter names to JSON Schema definitions (required).
    
    Usage:
        @mcp_tool(
            input_schema={
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            }
        )
        def my_tool(self, query: str):
            '''Tool description from docstring.'''
            return {"result": "..."}
    """
    def decorator(f: Callable) -> Callable:
        # Check whether the original function is a coroutine
        is_coroutine = inspect.iscoroutinefunction(f)
        
        if is_coroutine:
            @wraps(f)
            async def wrapper(*args, **kwargs):
                return await f(*args, **kwargs)
        else:
            @wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)
        
        # Build the complete inputSchema
        properties = {}
        required = []
        
        for param_name, param_schema in input_schema.items():
            if param_name == 'self':
                continue
            
            param_schema_copy = param_schema.copy()
            if "default" not in param_schema_copy:
                required.append(param_name)
            
            properties[param_name] = param_schema_copy
        
        input_schema_complete = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
        
        # Extract description from docstring
        description = inspect.cleandoc(f.__doc__) if f.__doc__ else ""
        
        # Build the full MCP schema
        wrapper._mcp_schema = {
            "name": f.__name__,
            "description": description,
            "inputSchema": input_schema_complete
        }
        wrapper._is_mcp_tool = True
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def collect_tool_schemas(agent_instance) -> List[Dict[str, Any]]:
    """Collect all internal tool schemas from an agent instance.

    Returns:
        List of tool schemas in MCP format.
    """
    schemas = []
    
    for name in dir(agent_instance):
        if name.startswith('_'):
            continue
        
        attr = getattr(agent_instance, name)
        
        if hasattr(attr, '_is_mcp_tool') and hasattr(attr, '_mcp_schema'):
            mcp_schema = attr._mcp_schema
            schemas.append({
                "name": mcp_schema['name'],
                "description": mcp_schema['description'],
                "inputSchema": mcp_schema['inputSchema']
            })
    
    return schemas
