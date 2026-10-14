---
name: optimizer-create-subagent
description: Universal sub-agent design — delegation mechanics, the canonical role archetypes (planner / executor / verifier / critic / synthesizer / specialist / memory), task-content and output-contract design, and the file-by-file checklist for adding a new agent. LOAD BEFORE adding a sub-agent. 
---

# Read this skill BEFORE adding a sub-agent

Adding a sub-agent is a **Level 4 action** — the highest level in the optimization action space. Before proceeding, confirm you have **escalation evidence**:

- **Direct structural justification**: trajectory analysis reveals context overload (`avg_phases ≥ 3.5`, high `avg_rounds`) or cognitive mode conflict that Level 1-3 actions cannot address by design, OR
- **Retrospective justification**: the same failure pattern persisted for 2+ steps despite Level 1-3 interventions (cite the hypothesis IDs and steps in your ledger).

Specifically, you need evidence for at least one of:
1. **Context overload** — agent uses many phases/rounds and context fills up. (Structural — only L4 provides fresh context.)
2. **Role conflict** — two distinct cognitive modes interfere in the same trajectory. (Structural — only L4 provides role isolation.)
3. **Repeated structured sub-task** — a clearly-defined sub-task recurs across queries, AND a Level 1-2 skill for the pattern was tried and proved insufficient.
4. **Low verification quality** — agent produces patches but fails tests, AND Level 1-2 verification skills were tried and proved insufficient due to confirmation bias.
5. **Task complexity** — the task naturally decomposes into phases that benefit from specialised roles, AND Level 3 prompt restructuring could not resolve the context pressure.

Record your escalation reasoning in the hypothesis ledger: which lower levels were considered, why they are insufficient, and what structural signal justifies Level 4.

This recipe ensures you add a sub-agent correctly. Follow it step by step and `finalize_optimization` will validate the result.

# Delegation mechanics — `assign_task_to_agent`

The parent agent delegates by calling:
```
assign_task_to_agent({
  "agent_name": "<sub-agent id, must be in parent's `subagent` list>",
  "task_content": "<self-contained natural-language task description>",
  "resume_previous": false   // true → reuse stored sub-agent + its memory
})
```
What happens:
1. The system validates the sub-agent id is in the caller's `subagent` list and is not an ancestor (cycle prevention).
2. A new (or resumed) sub-agent instance starts its own ReAct loop on `task_content`.
3. The sub-agent runs to `task_done`, returning its `task_summary` to the parent.
4. The parent's tool-call result is exactly that `task_summary`.

Critical properties (the dominant constraints on multi-agent design):
- The sub-agent has **its own** conversation history, context budget, system prompt, skills, and tool set. The parent sees nothing of its internal reasoning.
- `task_content` is the sub-agent's **only** view of the problem. Anything not in `task_content` is invisible to it. **Make `task_content` self-contained.**
- The parent receives only `task_summary`. **Design the output contract** explicitly in the sub-agent's system prompt and reinforce it in `task_content`.
- Each call blocks the parent until the sub-agent finishes.

# Two principles that govern good role design

1. **Minimum viable competency.** Each role does one thing well. Overlapping competencies between agents = duplication, not specialisation.
2. **Tool / skill set matches the role.** A role that does not need a tool should not have it. Smaller surfaces yield more reliable selection by the underlying LLM.

# Universal role archetypes

These archetypes are **task-domain-agnostic**. They apply to coding MAS, web research MAS, math reasoning MAS, GUI control MAS, multimodal MAS, etc. The specific tools and skills attached to each role are domain-dependent; the *shape* of the role is universal.

## Planner
- **Job**: decompose the user's task into an explicit sequence (or DAG) of sub-tasks; sequence them; produce a checklist the executor can follow.
- **Inputs**: original task + any high-level constraints.
- **Outputs**: structured plan (list of steps with explicit goals and success criteria for each).
- **Typical tool set**: minimal — usually only `read_file`, `write_file`, `task_done`, `phase_done_and_summarize`, `load_skill`. Often no domain tools at all.
- **When useful**: tasks with many sub-steps; tasks where the executor tends to drift without an explicit plan.

## Executor / Specialist
- **Job**: carry out one specific class of action with high reliability.
- **Inputs**: a single, well-scoped sub-task with explicit success criteria.
- **Outputs**: artefacts produced + a structured summary.
- **Typical tool set**: rich in the domain's primary tools (e.g., file editing tools for a code executor; search tools for a web-research executor; math computation tools for a math executor).
- **When useful**: when the action class recurs across queries and needs consistent execution discipline.

## Verifier
- **Job**: independently check whether a candidate solution actually satisfies the task. Critically, the verifier has not seen the executor's reasoning, so it does not inherit confirmation bias.
- **Inputs**: (task, candidate answer, supporting evidence).
- **Outputs**: confirm / refute + reasoning + (optional) better candidate.
- **Typical tool set**: read-only tools matching the executor's. No `write_file` on shared state.
- **When useful**: domains where wrong-but-confident answers are common; tasks where verification is meaningfully easier than generation.

## Critic / Reviser
- **Job**: take a draft output, identify weaknesses, propose a revised version.
- **Inputs**: (task, draft output, optional rubric).
- **Outputs**: revised output + diff explanation.
- **Typical tool set**: same as executor minus tools that change the world; can read but should not act on the environment.
- **When useful**: tasks with iterative refinement (writing, code review). Pair with an executor in a generate-critique-revise loop.

## Synthesizer / Aggregator
- **Job**: combine outputs from multiple sub-agents or multiple sources into a single coherent output that satisfies the original task contract.
- **Inputs**: multiple structured inputs + the original task.
- **Outputs**: final answer in the user-facing format.
- **Typical tool set**: minimal — file ops + analysis tools.
- **When useful**: when the task naturally decomposes into independent sub-problems whose results need integration.

## Memory / Librarian
- **Job**: maintain long-running structured state (knowledge graph, episodic log, persistent notes) that other agents query.
- **Inputs**: read or write requests from other agents.
- **Outputs**: queried fragments or write confirmations.
- **Typical tool set**: file ops + structured-extraction tools.
- **When useful**: long-horizon tasks where state must persist across many turns, or where multiple agents need a shared knowledge base.

## Investigator (analysis-only)
- **Job**: produce a structured findings report on a specific artefact (a trajectory, a document, a system state). Read-only by design.
- Useful as a sub-agent for deep analysis when the parent agent needs to focus on other work.

# Common topology recipes

These are universal *patterns*. The specific tools you assign depend on the task domain; the structure does not.

## Pattern A: Planner + Executor
```
entry: planner → executor (single specialist)
```
Useful when the task is clearly decomposable but execution is uniform.

## Pattern B: Executor + Verifier
```
entry: executor → verifier
```
Useful when wrong-but-confident answers are a dominant failure mode.

## Pattern C: Generator + Critic loop (delegation graph remains a DAG via fresh invocations)
```
entry: generator → critic (returns suggestions)
generator iterates internally based on critic output
```
Useful for iterative refinement tasks.

## Pattern D: Multi-specialist + Synthesizer
```
entry: planner → specialist_1, specialist_2, ... → synthesizer
```
Useful when sub-problems are independent and aggregation is non-trivial.

## Pattern E: Entry + Memory librarian
```
entry agent + memory librarian (queried as needed)
```
Useful for long-horizon tasks with persistent state.

# `task_content` design checklist

Every `assign_task_to_agent` call must include in `task_content`:
1. **The original task fragment** the sub-agent should solve (verbatim or precisely paraphrased).
2. **All relevant context** the parent has already established (facts, prior findings, constraints) — the sub-agent has no other access to them.
3. **The expected output format** — explicit schema or example.
4. **Hard constraints** — what the sub-agent must NOT do (helps a generic sub-agent stay in its lane).

A vague `task_content` is the single most common cause of useless sub-agent output. Treat it as a function call with required arguments.

# Output-contract design

In the sub-agent's `system.md`, include an explicit "Output contract" section that specifies what `task_summary` must contain. Examples:
- For a verifier: `{ confirmed: yes/no, better_candidate: <value or null>, reasoning: <text> }`.
- For a planner: a numbered list of steps with explicit success criteria for each.
- For an executor: produced artefacts (paths) + a summary of what was done.

Reinforce the contract in `task_content` ("Return your answer in the format specified by your output contract.").

# Step-by-step: adding a new sub-agent

All paths below are relative to `mas_config/`.

## Step 1 — copy a deployment block in `agents_llm_config.yaml`
Read the existing file. Copy the entire block from an existing agent under a new key. **Do NOT invent new model names or endpoints** — they reflect the physical deployment.
```yaml
agents:
  <existing_agent>:
    ...existing config...
  <new_agent_id>:
    llm_config:           # COPY VERBATIM from an existing agent
      model_name: "..."
      base_url: "..."
      max_retries: ...
    internal_tools_llm_config:    # COPY VERBATIM
      ...
    vlm_config:                   # COPY VERBATIM
      ...
    text_qa_chunk_chars: ...      # COPY (or raise for an extraction specialist)
    context_soft_limit_chars: ... # COPY VERBATIM
    context_hard_limit_chars: ...
    context_must_limit_chars: ...
    max_tool_result_chars_for_context: ...
```
The only safe value to change is `text_qa_chunk_chars` (e.g., raise for an agent specialising in long-document extraction). Never change context limits.

## Step 2 — add the agent to `config.yaml`
```yaml
agents:
  <existing_agent>:
    ...existing config (with subagent + assign_task_to_agent updates from Step 3)...
  <new_agent_id>:
    agent_name: "<new_agent_id>"
    is_entry_agent: false
    subagent: []                     # leave empty unless this sub-agent itself delegates
    max_iterations: ...              # match to role complexity
    max_real_iterations: ...
    prompt_dict:
      system: "./prompts/<new_agent_id>/system.md"
      initial_message: "./prompts/<new_agent_id>/initial_message.md"
    skill_dir: "./skills/<new_agent_id>"
    system_skills:
      - "manage-todo"
      - "phase-transition"
      - "handle-workspace"
    optional_skills: []              # add role-specific optional skills
    available_tools:
      # role-specific — see archetypes above and pick the minimal set.
      # Always include:
      - "read_file"
      - "write_file"             # only if the role writes anything
      - "list_file_tree"
      - "load_skill"
      - "task_done"
      - "phase_done_and_summarize"
      # plus the tools the role actually needs.
```

## Step 3 — wire delegation in the parent
In the parent's block, add the new agent to `subagent` and ensure `assign_task_to_agent` is in `available_tools`:
```yaml
  <parent_agent>:
    subagent:
      - "<new_agent_id>"
    available_tools:
      - ...existing tools...
      - "assign_task_to_agent"
```
Also update `prompts/<parent_agent>/system.md` to mention when the parent should delegate (otherwise the parent will keep doing the work itself).

## Step 4 — create prompt files
`prompts/<new_agent_id>/system.md` template:
```
You are `<new_agent_id>`, a specialist sub-agent. Your role: <one sentence>.

Today's date: ${today_date}

## Workflow
1. <Step 1>
2. <Step 2>
3. ...

## Output contract
When you call `task_done`, your `task_summary` MUST include:
- <field 1: type, semantics>
- <field 2: type, semantics>
- ...

Do NOT include speculation; only verifiable findings.

## Constraints
- <hard rule 1>
- <hard rule 2>

## System protocols
${system_protocols_str}

## Optional skills
${optional_skills_descp_str}
```

`prompts/<new_agent_id>/initial_message.md` — **CRITICAL RULES**:

The framework substitutes **exactly ONE placeholder**: `${task}`. This placeholder is replaced with the `task_content` string passed by the parent agent via `assign_task_to_agent`. **No other placeholders** (e.g., `${description}`, `${todo_details}`, `${reasoning}`) are supported — they will appear as raw `${...}` text at runtime.

The **correct and recommended** `initial_message.md` for any sub-agent is simply:
```
${task}
```

If you need structured content in the sub-agent's first message, the **parent agent** must compose everything into the `task_content` argument of `assign_task_to_agent`. The sub-agent's `initial_message.md` receives the fully-composed string via `${task}`.

**FATAL ERROR**: Writing an `initial_message.md` with custom placeholders like `${issue_description}` — these are NEVER resolved and cause the sub-agent to see raw placeholder text instead of actual content. `finalize_optimization` will reject templates without `${task}`.

## Step 5 — create skill directory
Create `skills/<new_agent_id>/` and at minimum copy the three system protocol skills referenced by `system_skills` (`manage-todo`, `handle-workspace`, `phase-transition`). Add role-specific optional skills as needed.

**CRITICAL SKILL STORAGE RULES:**
1. Each agent has its own **independent** `skill_dir` at `./skills/<agent_id>`. Set `skill_dir: "./skills/<new_agent_id>"` in `config.yaml`.
2. All skills for agent `<X>` MUST be placed under `skills/<X>/<skill-name>/SKILL.md`. Never place skill folders directly under `skills/` — the loader will not find them.
3. **Never reuse another agent's skill_dir.** For example, if agent `verifier` exists alongside agent `worker`, verifier's skill_dir MUST be `./skills/verifier`, NOT `./skills/worker` or `./skills`.
4. If two agents need the same skill content, **copy** the SKILL.md into each agent's skill directory independently. Do NOT share via path references.
5. Use **plain skill names** (no `/` prefix). For a skill called `framework-test` for agent `verifier`, the path is `skills/verifier/framework-test/SKILL.md` and the config lists `"framework-test"`.
6. Each SKILL.md must have valid YAML frontmatter with `name` (matching folder name exactly) and `description` fields.

## Step 6 — validate
Call `finalize_optimization`. It auto-checks all invariants. Fix any reported errors before `task_done`.

# Safety invariants (fatal if violated)

1. Exactly one entry agent (`is_entry_agent: true`).
2. No delegation cycles. The graph must be a DAG.
3. Full reachability — every non-entry agent appears in at least one `subagent` list.
4. Both YAML files agree — every agent in `config.yaml` has an entry in `agents_llm_config.yaml` and vice versa.
5. All file paths in `prompt_dict` and `skill_dir` resolve.
6. `assign_task_to_agent` requires non-empty `subagent` for it to do anything.
7. Every agent's `initial_message.md` **must** contain `${task}`. No other `${...}` placeholders are supported.
8. Each agent's `skill_dir` must point to `./skills/<agent_id>` — never to another agent's directory or a shared root.
9. All skills are referenced by **plain names** (no `/`). Each skill lives at `skills/<agent_id>/<skill-name>/SKILL.md`.

# Hypothesis discipline

Adding a sub-agent is one hypothesis. Record:
- **Level**: `L4` — with escalation reasoning (why Level 1-3 are insufficient for this root cause).
- **Observation**: which evidence in the batch motivated the split (cite query IDs and trajectory metrics).
- **Action**: "Add sub-agent <id> with role <archetype>; wire delegation in <parent>."
- **Expected outcome**: concrete, measurable — e.g., "<parent>'s `avg_rounds` drops by ≥30% because <work> is delegated; total accuracy does not drop in the next training batch."
- **Status**: `applied` (await empirical evidence; never `confirmed` at creation).

If the next batch shows accuracy regression, analyze the training trajectories to determine whether the sub-agent split caused the regression (e.g., poor task_content design, wrong role boundaries, or unnecessary delegation overhead). Pay special attention to **resampled queries** (re-included from the previous step's failures) — if they still fail after the sub-agent change, that is strong evidence the architecture change didn't help or hurt. Mark `refuted` and revert only if trajectory evidence confirms the sub-agent architecture caused concrete negative effects. Do NOT refute based on aggregate accuracy decline alone — each batch mixes resampled and freshly sampled problems.