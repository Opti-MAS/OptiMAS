# `write_file` — Create or overwrite / append to file

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | (required) | Relative to workspace or absolute. |
| `content` | string | (required) | The text to write. |
| `append` | bool | `false` | If true, appends to existing file. If false, overwrites. |

## Capabilities
- Creates parent directories automatically (`mkdir -p`).
- UTF-8 encoding only.
- Returns: `Successfully wrote/appended N characters (M lines) to <path>`.

## Also available: `write_file_lines`
An alternative that takes an array of strings (`content_lines`). Each element is one logical line; newlines are added automatically. Useful when escaping `\n` in a single string is error-prone.

## Limitations
- Overwrite mode (`append=false`) silently destroys existing content. No backup, no confirmation.
- No binary write support.
- `append=true` does NOT add a newline before the new content. If the existing file doesn't end with a newline, the first appended line will be glued to the last existing line.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `write_file` vs `file_edit` / `patch_file` | `write_file` does full-file creation or overwrite. `file_edit`/`patch_file` do targeted edits preserving unread content. Always prefer edit tools for existing files. |
| `write_file` → `read_file` | After writing, the agent can verify by reading back. |

## Common failure patterns
- Agent uses `write_file` to "edit" a config file → overwrites only remembered parts → destroys the rest. **Fix**: use `file_edit` or `patch_file` for existing files.
- Agent appends data without a leading `\n` → content glued to previous line. **Fix**: prepend `\n` in append content.
