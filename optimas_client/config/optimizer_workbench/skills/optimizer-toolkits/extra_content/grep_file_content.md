# `grep_file_content` — Search (and optionally replace) in files

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `pattern` | string | (required) | Search pattern. Literal by default; regex if `interpret_as_regex=true` or auto-detected. |
| `replacement` | string | `""` | If non-empty, performs search-and-replace across all matching files. |
| `path` | string | None | File or directory path. **Strongly recommended** — omitting searches entire workspace. |
| `lines_before` | int 0–20 | 3 | Context lines before each match. |
| `lines_after` | int 0–20 | 3 | Context lines after each match. |
| `max_results` | int 1–100 | 15 | Maximum matches returned. |
| `interpret_as_regex` | bool | `false` | Explicit regex mode. Auto-enabled when pattern contains `[]`, `\t`, `\d`, `+`, `?`, `.*`, `{N}`, etc. |
| `case_sensitive` | bool | `false` | Default is case-insensitive. |
| `include_glob` | string | None | Glob to include files (e.g., `"*.py"`). |
| `exclude_glob` | string | None | Glob to exclude files (e.g., `"*.min.js"`). |

## Implementation details
- **Backend**: ripgrep (`rg`) when available (fast, JSON streaming). Falls back to Python scanner if `rg` is not installed.
- **30-second soft timeout.** Search killed if exceeded; error suggests narrower scope.
- **Max file size: 10 MB.** Larger files skipped.
- **Broad-pattern guard.** Single-character patterns like `<`, `>`, `=` without `path` are rejected.
- **Auto regex promotion.** Obvious regex syntax auto-enables regex mode.

## Output format
Returns structured results. Each result: `file_path` with line number, `context` with `>>>` prefix for matching lines.

## Search-and-replace mode
When `replacement` is provided: all matching files modified in place. Regex captures (`\1`, `\2`) supported. **No undo.**

## Limitations
- Cannot search binary files (silently skipped).
- Replacement is immediate and global — no preview, no confirmation.
- Max 100 results. Summary note indicates total if more.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `grep_file_content` → `read_file` | Grep finds location; `read_file` retrieves context. Standard "locate then read" workflow. |
| `grep_file_content` (replace) vs `file_edit` | Grep-replace: batch pattern-based across files. `file_edit`: context-aware single-file. Use grep-replace for bulk renames; `file_edit` for structural changes. |
| `grep_file_content` vs `text_qa` | Grep is deterministic and instant; `text_qa` is LLM-powered and probabilistic. For exact strings, grep is always better. |

## Common failure patterns
- Agent searches entire workspace without `path` → slow, noisy, may timeout. **Fix**: always set `path`.
- Agent uses `replacement` on a broad pattern → unintended modifications. **Fix**: restrict `path` when using `replacement`.
