# `delete_file` — Delete file or directory

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `path` | string | (required) | File or directory path relative to workspace root. |

## Capabilities
- Deletes a single file (`unlink`) or entire directory tree recursively (`shutil.rmtree`).
- Path must exist; otherwise returns an error.

## Limitations
- **No undo.** Deleted content is permanently gone.
- **Recursive.** Deleting a directory removes everything inside it. No confirmation prompt.
- Path must be within the workspace boundary.

## Common failure patterns
- Agent deletes a directory containing files it needs later. **Fix**: skill enforcing "read before delete" or "move to `_trash/`" pattern.
