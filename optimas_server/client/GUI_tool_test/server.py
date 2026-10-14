"""
GUI Tool Test Server - FastAPI backend for tool testing interface.

Provides REST API endpoints for:
- Listing available tools
- Viewing tool schemas
- Calling tools for testing
"""
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from client.optimas_mcp_client import OptiMASClient

# Default configuration (from example_client.py)
DEFAULT_SERVER_URL = "http://127.0.0.1:8188/mcp"
DEFAULT_API_KEY = "optimas-local-dev-key"
DEFAULT_TASK_ID = "test-task-001"

# Global OptiMASClient instance
client: Optional[OptiMASClient] = None
client_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global client
    
    # Startup
    try:
        await ensure_client_connected()
        print(f"[INFO] Connected to MCP server: {DEFAULT_SERVER_URL}")
        print(f"[INFO] Session ID: {client.session_id}")
    except Exception as e:
        print(f"[ERROR] Failed to connect to MCP server: {e}")
        print("[WARN] Server will continue, but tool operations may fail")
    
    yield
    
    # Shutdown
    if client is not None:
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass
        client = None


app = FastAPI(
    title="GUI Tool Test Server",
    description="Backend server for testing MCP tools via web interface",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CallToolRequest(BaseModel):
    """Request model for tool call"""
    arguments: Dict[str, Any]


async def ensure_client_connected():
    """Ensure OptiMASClient is connected, create if necessary"""
    global client
    
    async with client_lock:
        if client is None or not client._connected:
            if client is not None:
                # Clean up existing client
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    pass
            
            # Create new client
            client = OptiMASClient(
                server_url=DEFAULT_SERVER_URL,
                api_key=DEFAULT_API_KEY,
                task_id=DEFAULT_TASK_ID
            )
            await client.__aenter__()
    
    return client


@app.get("/")
async def serve_index():
    """Serve the main HTML page"""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return {"message": "GUI not found. Please ensure static/index.html exists."}


@app.get("/api/tools")
async def list_tools():
    """List all available tools"""
    try:
        c = await ensure_client_connected()
        tools = await c.list_tools()
        
        # Format tools for frontend
        tools_list = []
        for tool in tools:
            tools_list.append({
                "name": tool.name,
                "description": tool.description or "",
            })
        
        return {"tools": tools_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@app.get("/api/tools/{tool_name}/schema")
async def get_tool_schema(tool_name: str):
    """Get schema for a specific tool"""
    try:
        c = await ensure_client_connected()
        schema = await c.get_tool_schema(tool_name)
        
        if schema is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return schema
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tool schema: {str(e)}")


@app.post("/api/tools/{tool_name}/call")
async def call_tool(tool_name: str, request: CallToolRequest):
    """Call a tool with provided arguments"""
    try:
        c = await ensure_client_connected()
        result = await c.call_tool(tool_name, request.arguments)
        
        if result.is_error:
            error_msg = "Tool call failed"
            if result.content and len(result.content) > 0:
                error_msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            
            return {
                "success": False,
                "error": error_msg,
                "data": None
            }
        else:
            return {
                "success": True,
                "data": result.data,
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/favicon.ico")
async def favicon():
    """Return 204 No Content for favicon requests to avoid 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    # Direct app object - simpler and more reliable
    # Note: reload won't work with direct app object, but it's more reliable
    uvicorn.run(app, host="127.0.0.1", port=21328, reload=False)

