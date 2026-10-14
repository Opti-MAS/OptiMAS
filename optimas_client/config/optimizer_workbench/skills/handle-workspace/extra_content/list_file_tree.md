### `list_file_tree` — Directory structure

**Parameters:** `path` (optional, default workspace root), `max_depth` (int 1–20, default 5), `follow_links` (bool, default false).

- Returns a tree-formatted string with directories (trailing `/`) and files.
- Use `path` to scope and `max_depth` to control depth.
- Symlinked directories appear in the tree but their contents are not expanded unless `follow_links=true`.
