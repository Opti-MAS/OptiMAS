---
name: phase-transition
description: Decide whether to continue work via phase_done_and_summarize or finish via task_done.
---

**Use `phase_done_and_summarize` when:**
- Work is incomplete, at a natural phase boundary, or approaching context/iteration limits.
- Before calling: update your TODO file AND write any important analysis or findings into working notes under `optimizer/` (e.g., `optimizer/analysis_notes.md`). These files on disk survive across phases — your conversation history does not.
- The system will automatically compress your current conversation and carry a detailed summary to the next phase. However, complex analysis and structured findings are better preserved in files you write yourself.

**Writing the `progress_report`:**
Your progress report plus the auto-compressed conversation summary are what the next phase receives. Be specific and operational:
1. **Files already read** and their key content (e.g., "Read `current_batch_trajectories.json`: 4 queries, 1 correct, 3 failed.")
2. **Changes already made** with file paths (e.g., "Updated `mas_config/prompts/worker/system.md`: added verification guidance.")
3. **Key analysis findings** — specific patterns, not vague summaries.
4. **Working notes written** — mention which files under `optimizer/` contain your detailed analysis so the next phase knows to read them.

**Writing `remaining_tasks`:** ordered, concrete steps the next phase should execute immediately.

**Finish with `finalize_optimization` when:**
- All planned changes are complete or dropped with an explicit reason.
- The TODO file and hypothesis ledger reflect the current evidence.

Successful finalization ends the step. It validates YAML, MAS structure, skills,
prompts and TODO completeness. `task_done` is an alias for the same mandatory
checks; it offers no alternate completion path. On failure, fix the reported
errors and retry while the iteration budget remains.

**Rule:** if in doubt, use `phase_done_and_summarize`. Premature `task_done` loses unfinished work. A well-written phase summary costs nothing and preserves everything.
