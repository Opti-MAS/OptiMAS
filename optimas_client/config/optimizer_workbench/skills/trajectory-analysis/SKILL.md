---
name: trajectory-analysis
description: Universal protocol for analysing MAS trajectories — anatomy of a trajectory, the audit-log-first evidence discipline, and a domain-agnostic taxonomy of abnormal behavioural patterns to look for.
---

# Why this skill exists

`text_qa` is itself an LLM call. Asking it "Why did the agent do X?" without first showing what the agent actually did is asking a different LLM to *guess* at the trajectory using only the chunk of JSONL it sees. That guess inherits the same biases as the original agent and cannot see earlier phases, the audit log, or the system prompt. It often hallucinates a plausible-sounding root cause that is not what actually happened.

**Rule: never call `text_qa` on a trajectory until you have a concrete observation to ground the question in.** Every `text_qa` question must reference a specific turn, tool call, argument value, or quoted reasoning fragment that you have already verified directly (via `read_file` / `grep_file_content`).

# Anatomy of a trajectory in this framework

Every MAS run produces, for each query, a directory:
```
<query_id>_<hash>/
  tool_call/
    tool_calls.jsonl                     # structured audit log (start here)
  working_trajectory/
    <agent>_<invocation>_<phase>.jsonl   # one file per (agent, invocation, phase)
  draft/                                 # files the agent wrote (often a scratchpad)
  resources/                             # external artifacts the agent fetched
```

## `tool_calls.jsonl`
One JSON object per tool call, with fields like `tool_name`, `arguments`, `success`, `duration_ms`, `result` (truncated preview). It is **short, structured, and complete**. It tells you the entire behavioural signature of a query without any LLM interpretation. **Always read it first.**

## `working_trajectory/*.jsonl`
The full ReAct loop, one line per event. Common line types:

| `type` / `role` | Content |
|---|---|
| `type: "tool_schema"` | First line — the tool catalog the agent saw |
| `role: "system"` | System prompt at conversation start |
| `role: "user"` | Task message (initial or post-phase) |
| `role: "assistant"` | Agent reasoning + `tool_calls` array. `_llm_duration_seconds` for latency. |
| `role: "tool"` | Tool result. `_tool_metadata.success`, `_tool_metadata.execution_seconds`. |

The filename pattern `<agent>_<invocation_id>_<phase>.jsonl` lets you reason temporally:
- `agent` — which agent (entry agent or sub-agent)
- `invocation_id` — hexadecimal identifier; same id = same invocation across phases
- `phase` — integer from 0, incremented every `phase_done_and_summarize`

## Batch report — `current_batch_trajectories.json`
The framework-computed summary: per-query outcome (correct / wrong-answer / agent-error / grader-error), gold answer, agent answer, grader reasoning, and `agent_stats` per agent (rounds, phases, tool counts, tool distribution, delegation counts, skill loads). **Most surface-level metrics are already here — do not re-derive them by re-reading trajectories.**

# Mandatory analysis order (per failing query)

For each query you triage, follow these steps in order. Do NOT call `text_qa` until step 4.

**Step 1 — Read the audit log directly.**
`read_file <query_id>_<hash>/tool_call/tool_calls.jsonl`. Scan for:
- The sequence of tool names — does it match the strategy you would expect?
- Argument patterns — repetition, near-duplication, escalating verbosity, etc.
- Failure markers in `result` (`error` fields, HTTP status codes, "0 succeeded" messages, empty data arrays).
- `duration_ms` outliers — blocked or degraded tools.

Write down concrete numbers in your `optimizer/analysis_notes.md` (e.g., "37 calls to tool A, 0 calls to tool B; 12 of A's arguments contained the pattern X").

**Step 2 — Read assistant reasoning at decision points only.**
You do NOT need to read full trajectories. Use `read_file` with line ranges for:
- The first assistant turn (initial plan).
- The turn that introduces the candidate / artefact that became the final answer.
- The turn just before `task_done` (final rationale).
- The last turn before each `phase_done_and_summarize` (handoff state).

Quote the relevant 1–3 sentences into your notes.

**Step 3 — Cross-reference with the batch report.**
`current_batch_trajectories.json` already gives `agent_stats` (rounds, phases, tool distribution) and `grader_response`. Use these for the macro picture before diving deeper.

**Step 4 — NOW you may call `text_qa`, with grounded questions only.**
A grounded question references a specific observation from steps 1–3:

| Ungrounded (forbidden) | Grounded (acceptable) |
|---|---|
| "Why did the agent fail?" | "The grader says the gold answer is `<G>` but the agent submitted `<A>`. Search this file for the substring `<G>` (case-insensitive). For each occurrence, report the turn, the role, and 200 chars of surrounding context." |
| "What did the agent verify before finishing?" | "Read assistant turns 48–54 (the turns immediately before the final `task_done`). Quote the verification list verbatim and mark which items had explicit evidence vs. unverified." |
| "Why did the agent pick X?" | "Find the first turn where `<X>` is mentioned. Quote the assistant reasoning at that turn and list every tool result the agent had received up to that point." |

The grounded variants extract evidence; the ungrounded ones beg the LLM to guess.

**Step 5 — Classify the failure (see taxonomy below).**

# Universal failure taxonomy (domain-agnostic)

These categories apply to any task type — code-writing, web research, math, GUI control, multimodal reasoning, etc. Each implies a different intervention.

| Class | Definition | Typical evidence signature |
|---|---|---|
| **Input-not-acquired** | The information / artefact required to solve the task was never obtained. | Tools returned no useful result, or the right tool was never called. |
| **Input-acquired-but-discarded** | Required information appeared in some tool result but the agent did not act on it. | Substring search of trajectory finds the answer in a tool result; agent's reasoning afterwards ignores or contradicts it. |
| **Input-acquired-but-misused** | Right information, wrong final action / output. | Agent reasoning quotes correct facts but produces an output that doesn't match the requested format or selects the wrong attribute. |
| **Premature termination** | Agent stopped before doing the work. | Very low rounds; `task_done` early; minimal tool diversity. |
| **Loop / fixation** | Agent repeats the same kind of action with no progress. | Same tool with near-identical arguments; same hypothesis revisited; output unchanged across many turns. |
| **Context overflow** | Agent runs out of context before finishing. | High phases (`avg_phases ≥ 3`) with repeated work across phases; long phase summaries. |
| **Tool failure** | The tool itself failed (infra issue, not agent issue). | Audit log shows `success: false`, HTTP error codes, "X failed after N attempts". |
| **Coordination failure** *(MAS only)* | Sub-agent received insufficient `task_content`, or returned output the parent could not use. | Sub-agent's first turn shows confusion; parent's reasoning after delegation does not reference the returned summary. |
| **Goal drift** | Agent strays from the original task. | Final answer addresses a different question than the user asked; intermediate work is on a related but distinct sub-problem. |
| **Verification failure** | Agent submitted without checking. | No explicit verification step; final reasoning admits uncertainty. |

A pattern that doesn't fit any of these is not yet diagnosed — keep looking before proposing an intervention.

# Universal abnormal-behaviour patterns

Look for these signatures in `tool_calls.jsonl` regardless of task domain. They are MAS-level pathologies, not task-specific issues.

| Pattern | Concrete signature | Likely root cause |
|---|---|---|
| **Mono-tool fixation** | One tool dominates (>70% of calls) and other available tools are unused. | Agent treats one tool as the only solution; system prompt / skills do not surface alternatives. |
| **Tool-output ignoring** | Tool call → result returned → next assistant turn does not reference the result. | Reasoning loop is not actually reading tool outputs (often: result too long, truncated, or low-signal). |
| **Argument micro-variation** | Same tool with arguments that differ by ≤20% characters. | Brute-forcing instead of changing strategy. |
| **Tool-output flooding** | One tool result occupies most of the agent's context. | Result not summarised; need a chunked / extractive sub-call. |
| **Phase-boundary amnesia** | After `phase_done_and_summarize`, agent re-does work from the previous phase. | Phase summary insufficient; or scratchpad / external memory not used. |
| **Delegation black-hole** *(MAS)* | Sub-agent invoked but parent does not act on its output. | `task_content` was vague; sub-agent's `task_summary` did not match parent's needs. |
| **Skill underuse** | `optional_skills` are loaded never or only after >20 turns. | Descriptions are too vague for the agent to recognise the trigger; or system prompt does not prompt skill loading. |
| **Premature finalization** | `task_done` called before any verification step; final output uses hedging language ("likely", "best guess"). | Workflow lacks a mandatory verification stage. |
| **Empty exploration** | High `list_file_tree` / `grep_file_content` count with no follow-up reads. | Agent inspects the workspace shape but cannot decide what to engage with. |

# Cross-trajectory pattern extraction

A pattern only justifies a MAS change if it is **systemic**:
- Observed in ≥2 queries with the same underlying mechanism, OR
- Observed in 1 query AND it aligns with a prior `applied`/`confirmed` hypothesis in the ledger.

For each pattern, record concretely:

| Field | What to fill in |
|---|---|
| Pattern name | A short label (e.g., "mono-tool fixation on tool X"). |
| Queries showing it | IDs + the specific turns / tool calls that demonstrate it. |
| Concrete evidence | Numeric counts, quoted reasoning, audit-log excerpts. |
| Failure class | One of the universal taxonomy entries above. |
| Existing hypothesis it fits | Hypothesis ID it refines or supersedes, if any. |
| Proposed intervention class | Skill / tool-set / prompt / sub-agent split / report-and-skip. |

A row with empty "Concrete evidence" is **not** a pattern — it is a guess.

# Mapping patterns to intervention classes

Use this matrix to choose the right rung of the intervention ladder (see `optimizer-mas-evolve`). The matrix is **domain-agnostic**.

| Pattern type | First-choice intervention | Skill to load |
|---|---|---|
| One specific failure mode that recurs across a query class | Create or strengthen an **optional skill** | `optimizer-skill-design` |
| A behaviour that should *always* happen | Add or extend a **system / preload skill** | `optimizer-skill-design` |
| Tool is being misused, ignored, or overused | Adjust `available_tools` and add usage guidance in the relevant skill | `optimizer-toolkits` |
| Two competencies in one agent interfere with each other | Split into a **sub-agent** with its own role | `optimizer-create-subagent` |
| Last resort — surface-level wording fix | Edit `prompts/<agent>/system.md` directly | (no skill — direct edit) |

Reaching for the system prompt first is the dominant failure mode of past optimization runs. Walk the ladder.

# Templates for grounded `text_qa` questions

Use these after steps 1–3 of the analysis order. Each references a specific artefact you have already inspected.
```
{
  "file_path": "<query_dir>/working_trajectory/<agent>_<inv>_<phase>.jsonl",
  "question": "Read assistant turns A through B. List, in order, every tool call and the first 200 characters of each result. Output as a markdown table: turn, tool, args_summary, result_summary."
}
```
```
{
  "file_path": "<query_dir>/working_trajectory/<agent>_<inv>_<last_phase>.jsonl",
  "question": "The grader says the gold answer is '<GOLD>'. Search the entire file (case-insensitive) for substrings of '<GOLD>'. For each match, report the turn number, role, and 200 chars of surrounding context."
}
```
```
{
  "file_path": "<query_dir>/working_trajectory/<agent>_<inv>_<last_phase>.jsonl",
  "question": "Quote the assistant's reasoning text on the turn that called `task_done`. Then, for each constraint or sub-question in the original task, mark whether the reasoning provides explicit evidence, hedged language, or no mention."
}
```

# Discipline reminders

- The audit log is your truth source; everything else is interpretation.
- A pattern with no audit-log evidence is a hypothesis about a hypothesis. Do not act on it.
- Tool failures (HTTP errors, infra issues, "X failed after N attempts" in the result) are **not** MAS optimization opportunities. Record them as such, do not patch the agent.
- The cost of a wrong intervention is one wasted step; the cost of waiting one step for more evidence is also one wasted step. When in doubt, gather more evidence.
