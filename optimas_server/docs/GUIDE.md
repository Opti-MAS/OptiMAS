# OptiMAS Server Usage Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Configuration](#configuration)
3. [Available Tools](#available-tools)
4. [Client Usage](#client-usage)
5. [Deployment](#deployment)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
cd optimas_server
pip install -r requirements.txt
```

### Fetch the WorkBench Benchmark

The WorkBench domain tools (calendar, email, analytics, project management, CRM) are
registered from a third-party repository that is not vendored here. From the project
root:

```bash
git clone https://github.com/olly-styles/WorkBench.git optimas_server/workbench
git -C optimas_server/workbench checkout --detach 42f1b6dc18986f47701ca83de2d63dd4f95c59a0
```

Use this pinned revision: later upstream versions change the sandbox API. Without a
compatible checkout, file tools remain available but WorkBench tasks cannot run.

### Configure

Copy the example configuration and edit it:

```bash
cp config.yaml config.local.yaml
```

At a minimum, set your API key and session secret:

```yaml
auth:
  api_keys:
    - "your-api-key-here"

session:
  secret_key: "your-secret-key-here"
```

Generate secure keys:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Start the Server

```bash
# Using default config.yaml
python -m src.main

# Using a custom configuration file
python -m src.main --config config.local.yaml
```

The server starts at `http://127.0.0.1:8188` by default.

### Verify

```bash
curl http://127.0.0.1:8188/health
```

---

## Configuration

The server uses a YAML configuration file. All sections are optional and have sensible defaults.

### Server

```yaml
server:
  host: "127.0.0.1"     # Listening address
  port: 8188             # Listening port
  debug: false           # Debug mode (disable in production)
```

### Authentication

```yaml
auth:
  api_keys:              # List of valid API keys
    - "key-1"
    - "key-2"
```

Clients authenticate by sending the key in the `X-MCP-Key` HTTP header.
`MCP_API_KEY` overrides the configured key list, and `OPTIMAS_SESSION_SECRET`
overrides `session.secret_key`. Keep these in the ignored `.env` file.

### Session

```yaml
session:
  timeout_seconds: 18000   # Session timeout in seconds (default: 5 hours)
  cleanup_interval: 300    # Expired session cleanup interval (seconds)
  max_sessions: 0          # Max concurrent sessions (0 = unlimited)
  secret_key: "..."        # HMAC key for session ID generation (keep secret!)
```

**Important**: Always use a strong random `secret_key` in production. Never commit it to version control.

### Logging

```yaml
logging:
  log_file: "logs/mcp_server.log"
  log_level: "INFO"              # DEBUG, INFO, WARNING, ERROR
  enable_file_logging: true
  enable_console_logging: true
```

### Workspace

```yaml
workspace:
  path: "../workspace"           # Relative to the server config file
```

Each session gets an isolated subdirectory under this path. Client and server must
use the same storage. Defaults share the repository-root `workspace/`; on separate
hosts use a shared filesystem.

### Dashboard

```yaml
dashboard:
  enabled: true                  # Serve /dashboard and its /api/* routes
  allow_remote: false            # Permit them on a non-loopback bind
```

The dashboard requires HTTP authentication. Send `X-MCP-Key` or use HTTP Basic
with any username and the MCP API key as password. Browser requests reuse those
credentials. Remote registration is opt-in through `allow_remote`; use TLS for
remote access. Set `enabled: false` to remove it entirely.

---

## Available Tools

### Session

| Tool | Description |
|------|-------------|
| `handshake` | Establish a session (must be called before any other tool) |
| `close_session` | Explicitly close the current session |

### File Operations

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (supports line range, rich file formats) |
| `write_file` | Write/overwrite file contents |
| `write_file_lines` | Write file by line range |
| `read_file_for_qa` | Read file in chunks for Q&A |
| `read_image_for_qa` | Read image as base64 data URI |
| `file_edit` | Edit file (insert, replace, delete lines) |
| `patch_file` | Apply a unified diff patch to a file |
| `replace_file_block` | Replace a block of text in a file |
| `grep_file_content` | Search file contents with regex (uses ripgrep) |

### System Operations

| Tool | Description |
|------|-------------|
| `list_file_tree` | List the file tree of the workspace |
| `delete_file` | Delete a file or directory |
| `move_file` | Move or rename a file/directory |

### WorkBench Tools

WorkBench domain tools (calendar, email, analytics, project management, CRM) are dynamically registered at startup from the pinned WorkBench checkout. Domain operations affect synthetic sandbox tables.

| Tool | Description |
|------|-------------|
| `workbench_reset_state` | Reset all WorkBench sandbox databases to baseline |
| `workbench_get_recorded_actions` | Retrieve recorded action strings from the current session |
| `calendar_*`, `email_*`, etc. | Domain-specific tools registered from WorkBench toolkit |

### Path Semantics

All file paths are **relative to the session workspace root**. Clients use paths like `subdir/file.txt` — the server resolves them within the session's isolated directory. Access outside the workspace is denied.

---

## Client Usage

### Using OptiMASClient (Recommended)

`OptiMASClient` handles connection, handshake, and session management automatically:

```python
from client.optimas_mcp_client import OptiMASClient

async with OptiMASClient(
    server_url="http://127.0.0.1:8188/mcp",
    api_key="your-api-key",
    task_id="my-task-001",
) as client:
    # Read a file
    result = await client.call_tool("read_file", {"file_path": "example.txt"})
    print(result)

    # Write a file
    await client.call_tool("write_file", {
        "file_path": "output.txt",
        "content": "Hello, world!",
    })

    # Search file contents
    result = await client.call_tool("grep_file_content", {
        "pattern": "TODO",
        "path": "src/",
    })
```

See [`client/example_client.py`](../client/example_client.py) for a complete example.

### Session Lifecycle

1. **Handshake**: Client calls `handshake` with a `task_id` to create a session
2. **Tool Calls**: Subsequent calls include the `session_id` (handled automatically by `OptiMASClient`)
3. **Expiration**: Sessions expire after `timeout_seconds` of inactivity
4. **Cleanup**: Expired sessions are removed by a background cleanup task

### GUI Tool Tester

A web-based tool tester is available for interactive exploration:

```bash
cd optimas_server/client/GUI_tool_test
python server.py
```

Then open `http://localhost:21328` in a browser to browse, inspect, and test all available tools.

---

## Deployment

### Development

```bash
python -m src.main --config config.yaml
```

### Production

The MCP endpoint and dashboard require API-key authentication. The dashboard grants
workspace administration and tool execution, so remote access requires appropriate
credentials and TLS. Non-loopback startup rejects the shipped development API key
and session secret.

1. **Disable debug mode**: Set `server.debug: false` (this also removes `/debug/routes`)
2. **Use strong keys**: Generate random API keys and a random `session.secret_key`
3. **Decide about the dashboard**: leave `dashboard.allow_remote: false` (recommended),
   or set `dashboard.enabled: false` to remove it entirely
4. **Use a process manager**. `src.main` exposes `create_app()` as a factory, not a
   module-level `app`, so pass `--factory`:

```bash
pip install "uvicorn[standard]"
uvicorn "src.main:create_app" --factory \
  --host 127.0.0.1 --port 8188 --workers 1
```

Terminate TLS at the reverse proxy. Use one worker: MCP sessions and WorkBench state
live in process memory. Async concurrency handles multiple tasks within that worker.
Match the bind address to `server.host`; prefer `python -m src.main --config
config.local.yaml` for a custom bind so startup checks use the actual configuration.

5. **Reverse proxy** (recommended — terminate TLS and add auth):

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8188;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

6. **Set appropriate log level**: Use `INFO` or `WARNING` in production

---

## Troubleshooting

### Server Fails to Start

**Port already in use**:
```bash
lsof -i :8188          # Find the process using the port
kill <PID>             # Or change server.port in config
```

**Missing dependencies**:
```bash
pip install -r requirements.txt
```

### Authentication Failure

- Verify the API key matches one in `auth.api_keys`
- Ensure the client sends it via the `X-MCP-Key` header
- Check server logs: `tail -f logs/mcp_server.log`

### Session Expired

Increase the timeout:
```yaml
session:
  timeout_seconds: 36000  # 10 hours
```

### Tool Call Failure

1. Ensure `handshake` was called first
2. Verify `session_id` is being passed (automatic with `OptiMASClient`)
3. Check tool call logs: `tail -f logs/mcp_server_tools.jsonl`
4. Review server logs for detailed error messages

### Log Locations

| Log | Path |
|-----|------|
| Server log | `logs/mcp_server.log` |
| Tool call log | `logs/mcp_server_tools.jsonl` |
