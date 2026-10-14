# `text_extract` — Structured multi-file extraction

## Parameters
| Param | Type | Default | Description |
|---|---|---|---|
| `file_paths` | list[string] | (required) | List of file paths to extract from. |
| `task` | string | (required) | Extraction task description (what to pull from each document). |

## How it works
1. Reads each file via MCP (`read_file` → fallback to `read_file_for_qa` for large files).
2. Splits large text into overlapping halo chunks (~50k chars core, 200-char halo overlap).
3. For each chunk, LLM call with structured extraction prompt → JSON output with dimensioned fields.
4. All chunk LLM calls run with bounded concurrency (up to 20 parallel).
5. Returns per-chunk results with `dimensions` (parsed JSON) and `raw_answer`.

## Output format
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "file_path": "...", "chunk_index": 0, "task": "...",
        "success": true, "raw_answer": "{ ... }",
        "dimensions": { "time": "...", "key_info": "..." },
        "metadata": { "chars_in_chunk": 45000 }
      }
    ]
  }
}
```

## Limitations
- **Expensive.** One LLM call per chunk per file. 5 files × 3 chunks = 15 LLM calls.
- **JSON output quality varies.** Model may produce malformed JSON; tool catches parse errors but does not retry.
- **Rich formats NOT supported** (unlike `text_qa`).
- **No cross-file synthesis.** Each chunk extracted independently. Agent must aggregate.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `text_extract` vs `text_qa` | `text_extract`: structured batch extraction (multiple files, JSON). `text_qa`: free-form single-file Q&A. Different use cases. |
| `text_extract` → post-processing | Agent receives raw JSON dimensions and must aggregate/compare/decide. |

## Common failure patterns
- Agent calls `text_extract` on one small file → overkill. **Fix**: use `read_file` + agent reasoning.
- Agent passes vague `task` → inconsistent dimensions across chunks. **Fix**: specify concrete extraction fields.
