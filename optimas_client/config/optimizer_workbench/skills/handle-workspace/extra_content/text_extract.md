### `text_extract` — Structured multi-file extraction

**Parameters:** `file_paths` (list of paths, required), `task` (string describing what to extract, required).

- Reads multiple files, splits into overlapping chunks, and runs structured JSON extraction via LLM on each chunk.
- Results include `dimensions` (parsed JSON) per chunk.
- Concurrent LLM calls (up to 20 parallel).

**Best practices:**
- Use for **batch triage** across 3+ files when you need structured, comparable output.
- For deep single-file Q&A, use `text_qa` instead.
- Write a concrete `task` with named output fields for consistent extraction.
