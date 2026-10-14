"""
Example client for OptiMAS Server.

Demonstrates:
- Simplified client usage with OptiMASClient class
- Automatic connection and handshake
- Tool calling and listing
- Tool schema retrieval
"""
import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.optimas_mcp_client import OptiMASClient


async def main():
    """Main client example"""
    server_url = "http://127.0.0.1:8188/mcp"
    api_key = "optimas-local-dev-key"
    task_id = "test-task-001"

    try:
        # Use OptiMASClient - automatically handles connection and handshake
        async with OptiMASClient(
            server_url=server_url,
            api_key=api_key,
            task_id=task_id
        ) as client:
            print("[OK] Connected and handshake completed")
            print(f"[INFO] Session ID: {client.session_id}")
            
            # Get sanitized tool schemas (meta fields removed)
            print("\n" + "-" * 60)
            print("Sanitized tool schemas from get_tool_schemas():")
            schemas = await client.get_tool_schemas()
            print(f"[OK] Retrieved {len(schemas)} tool schemas\n")
            for tool_name, schema in schemas.items():
                print(f"Tool: {schema['name']}")
                print(f"Description: {schema['description']}")
                if 'inputSchema' in schema:
                    print(f"inputSchema: {json.dumps(schema['inputSchema'], indent=2, ensure_ascii=False)}")
                print("-" * 60)
            
            # Call a tool (session_id is automatically included)
            print("\n" + "-" * 60)
            print("Calling tool: echo")
            result = await client.call_tool(
                "echo",
                {"message": "Hello from OptiMASClient!"}
            )
            
            if result.is_error:
                error_msg = "Tool call failed"
                if result.content and len(result.content) > 0:
                    error_msg = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                print(f"[ERROR] {error_msg}")
            else:
                print(f"[OK] Tool result: {result.data}")
    
    except Exception as e:
        print(f"[ERROR] Client operation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
