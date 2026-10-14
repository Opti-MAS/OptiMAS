# `list_file_tree` — Directory tree listing

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `path` | string | `""` (workspace root) | Subdirectory to list. |
| `max_depth` | int 1–20 | 5 | Maximum tree depth. 1 = only direct children. |
| `follow_links` | bool | `false` | If true, descend into symlinked directories. |

## Output format
Tree string with connectors (`├──`, `└──`, `│`). Directories have trailing `/`. Sorted: directories first, then files, alphabetically.

## Implementation details
- Walks directory recursively up to `max_depth`.
- Respects configurable ignore list (e.g., `__pycache__`, `.git`).
- Optionally hides agent's own `working_trajectory/` directory (server setting).

## Limitations
- **No file metadata** (size, mtime, permissions). Only names and tree structure.
- Large workspaces at high depth → very long output. Use `path` and `max_depth` to constrain.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `list_file_tree` → `read_file` / `grep_file_content` | Tree gives layout; read/grep gives content. Standard exploration workflow. |
| `list_file_tree` vs `grep_file_content` | Tree shows what exists; grep shows where something is mentioned. Don't grep for a filename — check the tree. |

## Common failure patterns
- Agent calls `list_file_tree` at root with `max_depth=20` → massive output floods context. **Fix**: use `max_depth=2` first, then drill into subdirectories.
- Agent calls `list_file_tree` repeatedly across phases → wastes turns. **Fix**: write directory structure to a note file on first exploration.
