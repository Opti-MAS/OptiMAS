# `text_qa` — LLM-powered Q&A over text files

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | (required) | File to query. Supports plain text AND rich formats (PDF, DOCX, PPTX, XLSX, CSV, HTML). |
| `question` | string | (required) | Natural-language question to answer based on file content. |

## How it works
1. **Input safety**: rejects binary/media files with clear error. Rich formats handled via separate extraction path.
2. **Probe phase**: reads first 50 lines to estimate file size.
3. **Chunking strategy**:
   - **Small files** (fits in one chunk ~16k chars): single LLM call.
   - **Large files**: sliding-window QA. Each chunk processed sequentially; each LLM call receives accumulated answer + current chunk + question. All chunks processed — no early termination.
   - **Very large / uneven files** (e.g., JSONL with huge first-line tool_schema): falls back to `read_file_for_qa` (~96k per chunk).
   - **Rich formats**: extracted text chunked by character count, same sliding-window.
4. **Auto-summary**: if final answer exceeds 8,000 chars, a compression LLM call reduces it.

## LLM involvement
- Uses agent's `internal_tools_llm` (separate from main reasoning LLM). Adds latency and cost.
- Chunk size configurable via `text_qa_chunk_chars` in `agents_llm_config.yaml` (default ~16k).

## Limitations
- **LLM-powered, not deterministic.** Subject to hallucination, miscomprehension, instruction drift. For exact string searches, use `grep_file_content`.
- **Sequential chunking.** 10-chunk file = 10 LLM calls = 60–120s wall time.
- **Cumulative answer degradation.** Earlier details may be compressed/dropped as window progresses.
- **Rich format quality varies.** Scanned PDFs produce empty text. HTML tables become pipe-separated text.
- **Cost per call.** Each `text_qa` = 1–N internal LLM calls. Prefer `read_file` or `grep_file_content` when answer location is known.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `text_qa` vs `read_file` | `read_file` returns raw text (cheap, deterministic). `text_qa` interprets via LLM (expensive, probabilistic). Use `read_file` when you know where the answer is. |
| `grep_file_content` → `text_qa` | Grep locates strings → then `read_file` for precision, or `text_qa` for broader comprehension. |
| `text_qa` vs `text_extract` | `text_qa`: one question per file. `text_extract`: structured multi-file extraction with JSON output. Use `text_extract` for batch triage; `text_qa` for deep single-file investigation. |
| `text_qa` on `*.jsonl` | Trajectory JSONL files often have huge first-line tool_schema. `text_qa` handles via `read_file_for_qa` fallback, but it's expensive. Prefer `grep_file_content` + `read_file` with line ranges. |

## Common failure patterns
- Agent calls `text_qa` before reading the file → vague questions → hallucinated answers. **Fix**: skill enforcing "read first, then ask grounded questions."
- Agent calls `text_qa` 10+ times on same file → massive cost. **Fix**: batch questions or use `read_file`.
- Agent uses `text_qa` on a 200-line config file → wasteful. **Fix**: prefer `read_file` for files < 500 lines.
- Agent treats `text_qa` answers as ground truth without verification. **Fix**: cross-reference with `grep_file_content` or `read_file`.
