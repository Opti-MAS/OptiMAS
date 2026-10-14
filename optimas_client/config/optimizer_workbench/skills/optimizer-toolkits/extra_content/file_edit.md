# `file_edit` — Line-based content replacement

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | (required) | Relative to workspace or absolute. |
| `old_lines` | string | (required*) | Lines to find and replace. Must match file content exactly (whitespace, indentation). |
| `new_lines` | string | (required) | Replacement text. |
| `pre_context` | string | None | Lines immediately before `old_lines` to narrow match. |
| `post_context` | string | None | Lines immediately after `old_lines` to narrow match. |
| `require_unique` | bool | `true` | If true, fail when 0 or >1 matches. If false, replace all. |

*At least one of `old_lines`, `pre_context`, or `post_context` must be provided.

## How matching works
1. Splits file and pattern into lines.
2. Constructs pattern: `pre_context + old_lines + post_context` (contiguous, no gaps).
3. Searches for this exact sequence in the file.
4. If found: replaces only `old_lines` (not context) with `new_lines`.

## Defensive line-number stripping
If no match found, the tool checks for `read_file`-style line-number prefixes (e.g., `   1 | content`). If detected AND stripped versions match, the edit proceeds. This prevents failures from agents copying line-numbered output into `old_lines`.

## Also available: `patch_file` (substring find-and-replace)
Works at **substring** level, not line level. Parameters: `old_str`, `new_str`, `require_unique`. On failure, provides fuzzy suggestion (best-matching block with similarity ≥ 50%). Also has line-number-prefix fallback.
- `file_edit`: replace entire lines or blocks. Better with context disambiguation.
- `patch_file`: replace substring within a line or across line boundaries. Better for surgical changes.

## Also available: `replace_file_block` (line-range replacement)
Replaces lines `[start_line, end_line]` (1-based, inclusive) with `new_block_content`. No pattern matching — just a range. Useful with known line numbers.

## Limitations
- **Whitespace sensitivity.** `old_lines` must match exactly, including indentation. A single space difference causes failure.
- **File must not have changed between read and edit.** If another process modified the file, match will fail.
- **Cannot insert at the start of an empty file** — use `write_file` instead.
- `require_unique=true` (default) fails if pattern appears >1 time. Use more context to disambiguate.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `read_file` → `file_edit` | Always read first, then edit. Set `include_line_numbers=false` to avoid gutter contamination. |
| `file_edit` vs `write_file` | `file_edit` preserves unread content. `write_file` overwrites everything. **Always prefer `file_edit`** for existing files. |
| `file_edit` vs `grep_file_content (replace)` | grep-replace does pattern-based changes across files. `file_edit` is single-file, context-aware. Use grep-replace for bulk; `file_edit` for precise edits. |

## Common failure patterns
- Agent copies line-numbered output into `old_lines` → match failure. Fallback handles most cases but wastes one turn. **Fix**: use `include_line_numbers=false` when reading for edit.
- Agent provides too little context → `require_unique` failure. **Fix**: include `pre_context`/`post_context`.
- Agent builds `old_lines` from memory instead of `read_file` → mismatch. **Fix**: enforce read-before-edit discipline.
