### `grep_file_content` — Search or search-and-replace in files

**Parameters:** `pattern` (required), `path` (recommended — file or directory), `interpret_as_regex` (bool, default false), `case_sensitive` (bool, default false), `max_results` (int, default 15), `lines_before`/`lines_after` (int, default 3), `replacement` (string, optional — enables search-and-replace), `include_glob` / `exclude_glob` (optional).

- Uses ripgrep (`rg`) when available — fast, handles large codebases.
- **Auto regex promotion**: patterns with obvious regex syntax (`[]`, `\d`, `.*`, `+`, etc.) are auto-promoted even without `interpret_as_regex=true`.
- **30-second timeout**: narrow `path` and use specific patterns if searches are slow.

**Best practices:**
- **Always set `path`** to a file or subdirectory — workspace-wide search is slow and noisy.
- Use simple keywords for literal search. Set `interpret_as_regex=true` for advanced patterns.
- Default is case-insensitive. Set `case_sensitive=true` when precision matters.
- When using `replacement`, always restrict `path` to avoid unintended changes across the workspace.
