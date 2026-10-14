# OptiMAS Server

MCP (Model Context Protocol) server built on FastMCP, providing multi-client connections, session management, tool execution, and resource access.

## Features

- Multi-client support with concurrent connections
- API Key authentication
- Session management via handshake tool (HMAC-SHA256 session IDs)
- Modular tool system, easy to extend
- MCP resource access
- Comprehensive tool call logging
- `OptiMASClient` wrapper for simplified client usage

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` to set API keys and session secret:

```yaml
auth:
  api_keys:
    - "your-api-key-here"
session:
  secret_key: "your-secret-key"
```

Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

> The server binds to `127.0.0.1` by default. The dashboard at `/dashboard` and its
> `/api/*` routes require an API key. Browsers use HTTP Basic authentication with
> the API key as password. They are served on loopback by default. See [Deployment](docs/GUIDE.md#deployment) before
> exposing the server on a network.

See the [Usage Guide](docs/GUIDE.md) for details.

### 3. Start the Server

```bash
python -m src.main
```

The server starts at `http://127.0.0.1:8188`.

### 4. Use the Client

Use `OptiMASClient` (handles connection, handshake, and session_id automatically):

```python
from client.optimas_mcp_client import OptiMASClient

async with OptiMASClient("http://127.0.0.1:8188/mcp", api_key="your-key") as client:
    result = await client.call_tool("read_file", {"file_path": "example.txt"})
    print(result)
```

See the [Usage Guide](docs/GUIDE.md) and [`client/example_client.py`](client/example_client.py) for more.

## Project Structure

```
optimas_server/
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── middleware/         # Middleware (auth, session, logging)
│   ├── tools/             # Tool modules
│   ├── resources/         # Resource modules
│   └── utils/             # Utility functions
├── client/                # Client library and examples
├── docs/                  # Documentation
├── config.yaml            # Configuration file
└── requirements.txt       # Dependencies
```

## Documentation

- **[Usage Guide](docs/GUIDE.md)** -- Configuration, tools, deployment, and troubleshooting

## Endpoints

- **MCP**: `http://127.0.0.1:8188/mcp`
- **Health check**: `http://127.0.0.1:8188/health`
- **API docs**: `http://127.0.0.1:8188/docs`

## Development

```bash
pip install -r requirements-dev.txt
```
