### `text_qa` — LLM-powered Q&A on text files

**Parameters:** `file_path` (required), `question` (required).

- Supports plain text AND rich formats (PDF, DOCX, PPTX, XLSX, CSV, HTML).
- Small files: single LLM call. Large files: sliding-window chunking — each chunk refines a cumulative answer. All chunks are processed.
- Rejects binary/media files (images, archives, executables) — use `image_qa` for images.
- Auto-summarizes answers exceeding 8,000 chars.

**Best practices:**
- Prefer `read_file` with line ranges when you know (or can guess) where the answer is — it's faster and deterministic.
- Ask **specific** questions, not vague ones. Vague questions yield vague LLM answers.
- For exact string matching, use `grep_file_content` instead — it's instant and deterministic.
