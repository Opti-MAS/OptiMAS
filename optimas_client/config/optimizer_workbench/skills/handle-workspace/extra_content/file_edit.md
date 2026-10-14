### `file_edit` — Edit file by content matching

**Parameters:** `file_path` (required), `old_lines` (string, the lines to replace — must match exactly), `new_lines` (string, the replacement), `pre_context` / `post_context` (optional, to disambiguate), `require_unique` (bool, default true).

- Builds a contiguous pattern: `pre_context + old_lines + post_context`. Finds it in the file and replaces only `old_lines` with `new_lines`.
- If `require_unique=true`, fails when 0 or >1 matches. Use more `pre_context` / `post_context` to disambiguate.
- Handles `read_file` line-number gutters defensively — if `old_lines` carries `   N | ` prefixes, they are stripped automatically as a fallback.

**Also available:**
- `patch_file` — substring-level find-and-replace (good for within-line edits). Returns a unified diff. Includes fuzzy suggestion on miss.
- `replace_file_block` — replaces lines by line-number range `[start_line, end_line]`. No pattern matching needed.

**Best practices:**
- Always `read_file` first (with `include_line_numbers=false`) to get exact current content before editing.
- Provide enough surrounding context in `old_lines` or use `pre_context` / `post_context` to ensure unique matching.
