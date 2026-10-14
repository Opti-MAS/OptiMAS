### `write_file` — Create, overwrite, or append to file

**Parameters:** `file_path` (required), `content` (required), `append` (bool, default false).

- Creates parent directories automatically.
- `append=true` adds content to the end of an existing file (no automatic newline — prepend `\n` if needed).
- `append=false` (default) **overwrites** the entire file.

**Best practices:**
- To modify an existing file, prefer `file_edit` or `patch_file` — they preserve content you haven't read. Use `write_file` only for creating new files or full rewrites.
