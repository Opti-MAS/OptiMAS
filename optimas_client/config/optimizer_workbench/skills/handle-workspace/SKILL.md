---
name: handle-workspace
description: Navigate, read, edit, and search files in the optimizer workspace.
---

All file paths are **relative to the session workspace root**.

**Workspace layout:**
- `mas_config/` — the MAS configuration you optimize (config.yaml, agents_llm_config.yaml, prompts/, skills/).
- `<query_id>_<hash>/` — per-query training artifacts. Each contains:
  - `tool_call/tool_calls.jsonl` — **structured audit log of every tool call** (name, args, success, duration, result preview). Read this FIRST when investigating a query — it's short, structured, and gives a complete behavioral signature without LLM interpretation.
  - `working_trajectory/<agent>_<inv4>_<phase>.jsonl` — full conversation logs (one file per agent invocation per phase).
  - `draft/`, `resources/` — files the agent wrote / fetched during the run.
- `optimizer/` — **your own working directory**. All your working files must go here. Contains:
  - `hypothesis_ledger.md` — **persistent hypothesis record** (survives across ALL steps/epochs). Your primary decision-making tool. See hypothesis-ledger protocol.
  - `optimization_history.json` — **persistent metrics history** (survives across ALL steps/epochs). Structured JSON with a `steps` array; each entry contains accuracy, val_delta, wall time, per-agent stats, and optimization summary.
  - `todo.md` — your TODO file for the current step (see manage-todo protocol).
  - Working notes (`analysis_notes.md`, `failure_patterns.md`, etc.) — write analysis findings here. Persist across phases; archived per step.
  - `previous_summary.md` — last step's optimization summary.
  - `step_<N>_working_log/` — archived working files from past steps (read selectively, not exhaustively).
  - `working_trajectory/` — your conversation logs (system-managed, do not modify).
- `current_batch_trajectories.json` — the batch report with per-query results and agent statistics.

**Persistent vs archived files:**
- `hypothesis_ledger.md` and `optimization_history.json` are **never archived** — they accumulate across all steps as your long-term memory.
- All other files under `optimizer/` are archived into `step_<N>_working_log/` at the end of each step.

**Write all your files under `optimizer/`.** Do NOT write to `draft/` or other top-level directories.

**Tools for reading and exploring:**
- `list_file_tree` — explore directory layout (use `path` and `max_depth`).
- `read_file` — read content; use line range for large files.
- `grep_file_content` — search by pattern across files.
- `text_qa` — ask questions about long files (auto-chunked). Use on JSONL trajectories.

**Tools for writing and editing:**
- `file_edit` — replace a block of lines. Preferred for targeted changes.
- `write_file` — create or overwrite a file. Use for new files or full rewrites.
- `grep_file_content` with `replacement` — pattern-based find-and-replace.

**Using historical working logs:** `step_<N>_working_log/` folders contain archived notes, analysis, and TODO from past steps. Use selectively — the optimization history and hypothesis ledger give you the overview without opening individual logs.

After **all** MAS config changes, call `finalize_optimization` to validate YAML and structural invariants.
