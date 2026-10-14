# `move_file` — Move or copy a file

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `old_path` | string | (required) | Source path. Must be a file (not directory). |
| `new_path` | string | (required) | Destination path. |
| `keep_old` | bool | `false` | If true, copy (`shutil.copy2`). If false, rename/move. |

## Capabilities
- Creates parent directories automatically.
- In move mode, fails if destination already exists (prevents accidental overwrite).
- In copy mode, overwrites are allowed.
- Preserves file metadata when copying (`copy2`).

## Limitations
- **Files only.** Cannot move/copy entire directories. To move a directory, move files individually or use `write_file` + `delete_file`.
- In move mode, destination must not exist. No merge or overwrite option.
