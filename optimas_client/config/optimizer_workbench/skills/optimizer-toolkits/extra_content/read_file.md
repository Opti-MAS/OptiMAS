# `read_file` — Read file content

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | (required) | Relative to workspace or absolute. |
| `include_line_numbers` | bool | `false` | Adds gutter like `   1 | content`. The prefix is NOT file content. |
| `start_line` | int ≥ 1 | 1 | Inclusive, 1-based. |
| `end_line` | int ≥ 1 | None | Inclusive. If omitted, reads to EOF. |

## Capabilities
- **Plain text**: UTF-8 files. Binary / non-UTF-8 → error.
- **Rich formats** (auto-detected by extension): PDF (via pdfplumber — extracts text + tables + embedded images), DOCX (paragraphs + tables), PPTX (slide-by-slide text + tables), XLSX/XLS (sheet-by-sheet cell grids), CSV (auto-dialect detection), HTML (visible text after stripping scripts/styles). Rich-format extraction runs in a thread pool to avoid blocking.
- `start_line` / `end_line` work on extracted text lines for rich formats too.
- Output is wrapped in `<file_content>...</file_content>` tags. A header line `Total lines in file: N` precedes the content.

## Hard limits
- **40,000-character output cap.** If exceeded, output is truncated with a note. The file is NOT fully returned.
- Rich-format extraction also caps at 40k chars.

## Limitations
- Cannot read binary files (images, executables, archives). For images, use `image_qa`.
- When `include_line_numbers=true`, agents sometimes copy the gutter into `file_edit`/`patch_file`, causing match failures. A defensive fallback strips these prefixes, but use `include_line_numbers=false` when planning to edit.
- For very large files (>40k chars), output is silently truncated. Use `start_line`/`end_line` for specific ranges, or `text_qa` for QA.
- PDF extraction quality depends on pdfplumber; scanned PDFs with no embedded text return empty pages. Use `image_qa` on extracted images instead.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `read_file` vs `text_qa` | `read_file` returns raw text (cheap, no LLM). `text_qa` runs LLM on chunks (expensive, handles arbitrarily large files). Use `read_file` for targeted ranges; `text_qa` when answer location is unknown. |
| `read_file` vs `grep_file_content` | `grep_file_content` finds *where*; `read_file` retrieves surrounding content. Workflow: grep → locate line → read_file with range. |
| `read_file` → `file_edit` / `patch_file` | Read first to get exact content, then edit. Set `include_line_numbers=false` when feeding content to edit tools. |
| `read_file` vs `read_file_for_qa` | `read_file_for_qa` is internal to `text_qa`. Agents should not call it directly. |

## Common failure patterns
- Agent reads entire large file without line ranges → truncation → misses content. **Fix**: skill enforcing `start_line`/`end_line` on files > 500 lines.
- Agent uses `read_file` on a 10,000-line JSONL trajectory → context overflow. **Fix**: use `text_qa` or `grep_file_content` instead.
- Agent reads rich format but ignores `[Table N]` sections. **Fix**: skill reminding that tables are inline in extracted text.
