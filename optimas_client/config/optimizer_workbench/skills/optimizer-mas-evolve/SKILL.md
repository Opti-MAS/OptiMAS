---
name: optimizer-mas-evolve
description: Universal MAS architecture theory — what you can and cannot edit, single-vs-multi-agent trade-offs, when and how to evolve architecture, and the safety invariants enforced by `finalize_optimization`. LOAD BEFORE adding, removing, or rewiring agents.
---

# What a MAS is, in this framework

A Multi-Agent System (MAS) is a directed acyclic graph of agents. Exactly one agent is the **entry agent** — it receives the user's task and returns the final output. Each agent has its own:
- **System prompt** (role definition, workflow, constraints).
- **Skill set** (system / preload / optional bodies).
- **Tool set** (`available_tools`).
- **Conversation context** (independent from other agents).
- **Iteration budget** (`max_iterations` per phase, `max_real_iterations` total).

Agents communicate exclusively via the `assign_task_to_agent` tool: the parent sends a `task_content` string, the child runs its own ReAct loop, and the child's `task_summary` is returned to the parent as the tool result.

Importantly: an agent's reasoning, intermediate tool calls, and conversation history are **never visible** to other agents. The only inter-agent channel is the (`task_content`, `task_summary`) pair. This is the dominant constraint on multi-agent design.

# Single-agent vs multi-agent trade-offs

## Single-agent strengths
- **Simplest mental model.** No coordination overhead, no `task_content` design required.
- **Full context coherence.** All facts, intermediate results, and reasoning live in one conversation. No information loss across agent boundaries.
- **Cheapest per-step cost.** No delegation overhead, no second system prompt to load.

## Single-agent weaknesses
- **Context exhaustion.** As tasks get longer, the single conversation runs out of room. Phases compress but information is lost.
- **Role conflict.** When the same conversation must do creative exploration AND rigorous verification AND output formatting, the agent often does each poorly because each requires a different mindset.
- **Prompt saturation.** Adding workflow rules for every situation makes the system prompt long; long prompts get skimmed.

## Multi-agent strengths
- **Context isolation.** Each sub-agent gets a fresh context window. Total reasoning capacity scales linearly with number of agents.
- **Specialisation.** Each agent has a tight role, tool set, and skill bundle matched to its competency. Smaller surfaces are easier to optimize.
- **Independent verification.** A verifier sub-agent that has not seen the parent's reasoning is less prone to confirmation bias.
- **Reusability.** A well-designed sub-agent (e.g., a "verifier" or "extractor") can be invoked from many parents in more complex topologies.

## Multi-agent weaknesses
- **Coordination cost.** Every delegation = one extra LLM invocation, one full system-prompt of the sub-agent loaded, and the round-trip wait time.
- **Information loss at boundaries.** The parent never sees the sub-agent's intermediate reasoning. If `task_content` was incomplete, the sub-agent works on the wrong problem.
- **Output-contract failure.** If the sub-agent's `task_summary` does not match the parent's expected schema, the parent cannot use it.
- **Debugging cost.** Failures span multiple trajectory files; root cause attribution is harder.

**Default starting point** is typically a single agent. The MAS may already be single-agent or multi-agent when you begin optimizing — your job is to evolve it in whichever direction the evidence supports.

# When to evolve from single-agent to multi-agent

Evolving to multi-agent is a **Level 4 action** — the highest-impact intervention in the optimization action space. It requires stronger evidence than Level 1-3 actions, but is the right choice when the evidence supports it.

**Structural signals that justify Level 4 (direct escalation):**

1. **Context overload.** Per-query `avg_phases ≥ 3.5` AND `avg_rounds` approaches `max_iterations`. The agent is doing too many distinct things to keep them all in context. A sub-agent provides a fresh context window — Level 1-3 cannot solve this.
2. **Distinct competencies interfere.** Trajectory analysis shows the agent alternates between two cognitive modes that need different mindsets — e.g., open-ended exploration vs. rigorous verification. Splitting into specialised roles eliminates this interference — skills cannot provide context isolation.

**Signals that may justify Level 4 after lower levels have been tried:**

3. **Repeated structured sub-task.** A clearly-defined sub-task recurs across queries (e.g., "reproduce the bug and verify the fix by running tests"). First try Level 1-2 (skill to standardize the workflow). If 2+ steps show the skill is insufficient (the task is too context-heavy for skill-guided single-agent execution), escalate to Level 4.
4. **Verification quality is low.** The agent produces patches but tests still fail. First try Level 1-2 (verification-focused skills). If trajectory analysis shows the agent cannot verify its own work due to confirmation bias, escalate to Level 4 (independent verifier).
5. **Task complexity exceeds single-agent capacity.** The task naturally decomposes into phases that exceed the single agent's context. First try Level 3 (prompt restructuring). If context exhaustion persists, escalate to Level 4.

**If none of (1)–(5) hold**, Level 1-3 actions are sufficient. Do not add agents "just in case" — each agent adds coordination overhead, information loss at boundaries, and debugging cost.

When you do split, prefer a **flat topology**: entry → 1-3 specialists. Avoid A → B → C chains because coordination cost multiplies and root-cause attribution becomes much harder.

# MAS config layout
```
mas_config/
  config.yaml               # primary config — you edit this
  agents_llm_config.yaml    # deployment config — DO NOT edit (except when adding agents)
  prompts/<agent_id>/       # system.md, initial_message.md per agent
  skills/<agent_id>/        # one folder per skill, each with SKILL.md
```
`config.yaml` and `agents_llm_config.yaml` are **merged at load time per agent id**; `config.yaml` overrides on overlap. Both files MUST contain an entry for every agent id; missing entries cause load failures.

## What you CANNOT edit
`agents_llm_config.yaml` contains **deployment-specific settings** tied to the physical model deployment:
- `llm_config` (model_name, base_url, max_retries, read_timeout)
- `vlm_config`, `internal_tools_llm_config`
- `text_qa_chunk_chars`
- `context_soft_limit_chars`, `context_hard_limit_chars`, `context_must_limit_chars`
- `max_tool_result_chars_for_context`

Changing these causes connection failures or context overflow. The ONLY exception: when adding a new agent, you must copy a complete deployment block from an existing agent (see `optimizer-create-subagent`).

## What you CAN edit (the optimization surface)
In `config.yaml`, per agent:
- `max_iterations`, `max_real_iterations` — iteration budgets.
- `system_skills`, `preload_skills`, `optional_skills` — skill management (see `optimizer-skill-design`).
- `available_tools` — capability surface (see `optimizer-toolkits`).
- `subagent` — delegation graph edges.
- `is_entry_agent` — exactly one agent is `true`.

Content files:
- `prompts/<agent_id>/system.md` — agent system prompt. Highest token-cost edit.
- `prompts/<agent_id>/initial_message.md` — first user message template. **MUST contain `${task}`** — this is the ONLY supported placeholder. See critical rules below.
- `skills/<agent_id>/<skill-name>/SKILL.md` — skill content. Must be under the agent's own `skill_dir`.

### Critical: `initial_message.md` template rules

The framework resolves exactly ONE placeholder in `initial_message.md`: **`${task}`**. This is replaced at runtime with the `task_content` string from the parent's `assign_task_to_agent` call (for sub-agents) or the benchmark task (for entry agents).

**No other `${...}` placeholders are supported.** Placeholders like `${issue_description}`, `${patch_diff}`, `${reasoning}` will NOT be resolved and will appear as raw text. If the sub-agent needs structured input, the PARENT agent must compose everything into `task_content`.

The correct `initial_message.md` for any sub-agent is: `${task}` (just that, on one line). `finalize_optimization` rejects templates without `${task}`.

### Critical: `skill_dir` isolation rules

Each agent MUST have `skill_dir: "./skills/<agent_id>"`. This is the root for all skill lookups. Never set one agent's skill_dir to another agent's directory or to the shared `./skills` root. If two agents need the same skill, copy it independently into each agent's skill directory.

Topology:
- Add or remove agents (see `optimizer-create-subagent`).
- Rewire `subagent` lists.

# Optimization action space — leveled by scope and reversibility

When you have an evidence-backed hypothesis, choose the **lower-level action whose scope matches the diagnosed root cause**. Escalate only when evidence justifies it (see escalation principles in system prompt).

| Level | Action | Trade-offs | Best for |
|---|---|---|---|
| 1 | **Edit an existing skill** | Lowest blast radius — one file, conditionally activated | Refining or extending a strategy already in the MAS |
| 1 | **Remove/demote an unused skill** | Reduces context overhead, removes stale/contradictory guidance | Skills with 0 loads over 2+ steps, refuted hypotheses, or redundant coverage |
| 2 | **Create a new optional skill** | Low risk — one new file, on-demand load; but skill needs a clear description to be loaded | A **genuinely novel** failure pattern with no existing skill covering it |
| 2 | **Adjust `available_tools`** | Changes every LLM call's schema; may affect unrelated behaviours | Agent ignores a useful tool, overuses one, or lacks a needed capability |
| 3 | **Promote skill to system/preload** | Adds to every prompt; may cause instruction skimming if overused | A behaviour now needed on every query (proven by evidence) |
| 3 | **Edit the system prompt** | Global and persistent; repeated edits dilute prior instructions | Behaviour that must be unconditional AND cannot be expressed as a skill |
| 4 | **Add a sub-agent** | Highest structural impact: delegation overhead, coordination complexity, but fresh context windows and role specialisation. Follow `optimizer-create-subagent` for the structured recipe. | Context exhaustion, cognitive mode conflict, repeated structured sub-task, independent verification need |

### Escalation discipline

- **Parsimony.** If a Level 1-2 action can address the root cause, use it — even if a higher-level action would also work.
- **Evidence-driven escalation.** To justify Level N+1, you need: (a) the same failure pattern persisted for 2+ steps despite Level ≤N interventions, OR (b) trajectory analysis reveals a structural problem that Level ≤N cannot address by design (e.g., `avg_phases ≥ 3.5` → context exhaustion → only Level 4 provides a fresh context window).
- **No mandatory staircase.** You are NOT required to try every lower level first. If the evidence clearly points to a structural problem, jump directly — but record why lower levels are insufficient in the hypothesis ledger.
- **Prompt/skill bloat → stop adding Level 3.** If combined system prompt + preload text exceeds ~3000 tokens, further Level 3 additions are counterproductive. Consider Level 1-2 (restructure) or Level 4 (distribute work).

**Stopping rule for trivial patches.** If your last 2 hypotheses both ended in a Level 3 action and the same pattern recurred, do NOT repeat Level 3. Drop to Level 1-2 or escalate to Level 4 with evidence.

## Skill management discipline — MODIFY before CREATE, REMOVE when stale

**Strong bias: the MAS should NOT accumulate skills like patches.** Every skill has a maintenance cost: preload skills burn context tokens every phase, optional skill descriptions expand the system prompt, and overlapping skills create contradictory guidance.

**Before creating ANY new skill, check:**
1. Does an existing skill already address the same root cause? → **Edit it** instead.
2. Would a 2–3 line addition to an existing skill cover this? → **Extend it** instead.
3. Is the failure pattern truly systemic (≥2 queries)? → If not, do NOT create a skill.

**Before finalizing ANY step, review skill inventory:**
1. Are there skills that were never loaded (check `agent_stats.skill_calls`)? → Remove them from config.
2. Are there preload_skills that are only relevant to rare cases? → Demote to optional.
3. Are there system_skills with domain-specific content? → Demote to preload or optional.
4. Are total preload bodies concise (≤800 words combined)? → If not, trim or demote.
5. Do two or more skills overlap in scope? → Merge them.

**For specific recurring errors (patch-style skills):**
- These belong in `optional_skills`, never in system_skills or preload_skills.
- They must have highly specific `name` and `description` so the agent knows exactly when to load them.
- Example: `name: "fix-django-model-metaclass-ordering"`, `description: "Load when fixing bugs in Django model metaclass __new__ or field ordering — the __init_subclass__ hook has ordering constraints that naive patches break."`

# Safety invariants (validated by `finalize_optimization`)

1. Top-level key in `config.yaml` is `agents`.
2. Each agent key equals its `agent_name` value.
3. Exactly one agent has `is_entry_agent: true`.
4. No cycles in the delegation graph (it must be a DAG).
5. Every non-entry agent appears in at least one agent's `subagent` list (no dead agents).
6. `prompt_dict` paths point to real files; `skill_dir` points to a real directory.
7. Both `config.yaml` and `agents_llm_config.yaml` contain an entry for every agent.
8. Every agent's `initial_message.md` contains `${task}`. No other `${...}` placeholders are supported.
9. Each agent's `skill_dir` is `./skills/<agent_id>` — never another agent's directory or the shared root.
10. Skill names are plain (no `/`). Each skill at `skills/<agent_id>/<skill-name>/SKILL.md`.
11. No orphan skill folders at `skills/` root that aren't agent directories.

If `finalize_optimization` fails, fix the listed errors and call it again before `task_done`.

# Universal optimization principles

These are general inductive biases that hold across task domains.
- **Clarity over length.** A concise, well-structured prompt outperforms a verbose one. Agents follow structure; they skim prose.
- **Constraints prevent errors.** Negative constraints ("do NOT do X") are often more impactful than positive encouragements ("try to do Y") for correcting failure patterns.
- **Skills > prompt edits when conditional.** If a fix only matters for some query types, make it a skill.
- **Modify > Create.** Before adding a new skill, always check whether an existing skill can be extended. Skill proliferation without curation degrades the MAS.
- **Remove stale skills.** Unused skills (0 loads over 3+ steps) should be actively removed. A lean skill set is more effective than a bloated one.
- **Preload and system skills must be concise.** These are injected on every turn or phase. Keep each body ≤300 words, total preload ≤800 words. Use checklists, not prose.
- **Minimal capability surface.** Every tool you grant adds schema tokens to every LLM call. Pair every addition with a removal when possible.
- **One change per hypothesis.** Targeted, isolated changes are attributable. Multiple simultaneous changes make regression analysis impossible.
- **Investigate when validation regresses.** A negative validation Δ is an important warning (the val set is fixed), but it alone does not prove a change was harmful. Analyze training batch trajectories for concrete evidence of regression before reverting. Revert only when trajectory analysis confirms the change caused negative effects.
- **Evidence before action.** Every change traces to a hypothesis that traces to concrete observations from `tool_calls.jsonl` or specific trajectory turns. No intervention without evidence.
- **Preserve what works.** The `confirmed` hypotheses are the MAS's load-bearing insights. Subsequent edits should not silently dilute them; check before editing files those hypotheses touched.