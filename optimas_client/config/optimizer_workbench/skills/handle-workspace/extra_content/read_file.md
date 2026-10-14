### `read_file` — Read file content

**Parameters:** `file_path` (required), `include_line_numbers` (bool, default false), `start_line` (int ≥ 1, default 1), `end_line` (int, optional).

**Supports:** plain text (UTF-8) and rich formats — PDF, DOCX, PPTX, XLSX/XLS, CSV, HTML/HTM. Rich formats are auto-detected and text is extracted automatically.

**Output:** content wrapped in `<file_content>...</file_content>` tags, preceded by `Total lines in file: N`.

**Hard limit:** 40,000 characters. Longer files are truncated with a note. Use `start_line`/`end_line` to read specific ranges, or `text_qa` for Q&A on large files.

**Best practices:**
- Always use `start_line`/`end_line` on unknown or large files. Start with a small range (e.g., 1–80), then jump to the needed region.
- Set `include_line_numbers=false` (the default) when you plan to use the output in `file_edit` or `patch_file`.
- For long files where you need a specific answer, prefer `text_qa` over full-file reads.
- For images, use `image_qa` instead — `read_file` cannot interpret image content.
