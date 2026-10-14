# GUI Tool Test

A web-based GUI for testing and exploring MCP server tools.

## Features

- **Tools List**: Browse all available tools with search functionality
- **Schema Viewer**: View detailed input schemas for each tool
- **Tool Tester**: Test tools with a dynamic form generated from their schemas

## Requirements

- Python 3.8+
- FastAPI
- uvicorn
- OptiMASClient (from parent directory)

## Installation

The GUI uses the OptiMASClient from the parent directory, so no additional installation is needed if you're running from the `optimas_server` directory.

## Usage

### Start the GUI Server

```bash
cd optimas_server/client/GUI_tool_test
python server.py
```

Or using uvicorn directly (from the GUI_tool_test directory):

```bash
cd optimas_server/client/GUI_tool_test
uvicorn server:app --host 127.0.0.1 --port 21328
```

### Access the GUI

Open your browser and navigate to:

```
http://localhost:21328
```

## Configuration

The server uses default configuration from `example_client.py`:

- **Server URL**: `http://127.0.0.1:8188/mcp`
- **API Key**: `optimas-local-dev-key`
- **Task ID**: `test-task-001`

To change these settings, edit the constants in `server.py`:

```python
DEFAULT_SERVER_URL = "http://127.0.0.1:8188/mcp"
DEFAULT_API_KEY = "optimas-local-dev-key"
DEFAULT_TASK_ID = "test-task-001"
```

## Architecture

- **Backend** (`server.py`): FastAPI server that provides REST API endpoints
- **Frontend** (`static/`): Pure HTML, CSS, and JavaScript
  - `index.html`: Main page structure
  - `styles.css`: Styling and layout
  - `app.js`: Frontend logic and API calls

## API Endpoints

- `GET /api/tools` - List all available tools
- `GET /api/tools/{tool_name}/schema` - Get tool input schema
- `POST /api/tools/{tool_name}/call` - Call a tool with arguments

## Notes

- The GUI automatically connects to the MCP server on startup
- Session management is handled automatically by OptiMASClient
- All tool schemas are sanitized (meta fields removed) for LLM compatibility

