# Workflow control tools

## `load_skill`
- **What it does**: loads the body of an `optional_skill` into conversation as a tool result.
- **Required if**: agent has any `optional_skills` configured. Without `load_skill` in `available_tools`, optional skills are inaccessible.
- **Limitation**: loading costs context tokens. Only load when trigger conditions described in the skill's `description` are met.

## `task_done`
- **What it does**: marks agent's task as complete. `task_summary` is returned to caller.
- **Behavior**: ends the ReAct loop immediately. No further tool calls possible.
- **For sub-agents**: `task_summary` is the **only** information the parent receives. Everything else is invisible.
- **Limitation**: premature `task_done` → work lost and cannot be resumed (unless `resume_previous=true` used later).

## `phase_done_and_summarize`
- **What it does**: ends current phase, auto-compresses conversation, starts new phase with compressed context + phase summary.
- **Behavior**: agent provides `summary` describing accomplishments and next-phase guidance. The system:
  1. Compresses entire conversation via LLM (preserves key facts, drops verbose tool results).
  2. Injects compressed context + summary into new phase.
  3. Re-injects `preload_skills` (so they survive compression).
  4. Resets per-phase iteration counter.
- **When to use**: approaching per-phase iteration limit, or distinct task focus shift warranting clean context.
- **Limitation**: compression is lossy. Write important findings to files before calling.

## `assign_task_to_agent`
- **Parameters**: `agent_name` (must be in `subagent` list), `task_content` (self-contained task), `resume_previous` (bool, default false).
- **What it does**: delegates to sub-agent. Sub-agent runs its own ReAct loop on `task_content`, returns `task_summary`.
- **Behavior**: blocks parent until sub-agent completes. Parent sees only `task_summary`.
- **Limitation**: requires non-empty `subagent` list. `task_content` is sub-agent's ONLY view of the problem.

## `finalize_optimization` (optimizer-specific)
- **What it does**: validates MAS config — YAML syntax, file paths, agent graph (DAG, no orphans, one entry agent), consistency between both YAML files.
- **When to use**: after all MAS config changes, before `task_done`.
- **Behavior**: returns validation errors or success. Fix errors and retry if needed.

## Synergies and overlaps
| Pair | Relationship |
|---|---|
| `phase_done_and_summarize` → `load_skill` | After compression, optional skill bodies are lost. Re-load if needed. `preload_skills` are auto-re-injected. |
| `assign_task_to_agent` → `task_done` | Sub-agent's `task_done` returns `task_summary` to parent as the tool result. |
| `finalize_optimization` → `task_done` | Always call `finalize_optimization` before `task_done` in the optimizer. Failed validation + `task_done` leaves MAS broken. |
