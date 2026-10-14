---
name: optimizer-skill-design
description: Universal skill-design theory — file structure, the three skill categories (system / preload / optional), formatting rules, and a domain-neutral worked example for converting a recurring pattern into a loadable skill.
---

# Why skills are the highest-leverage extension mechanism

Editing a system prompt has three structural problems:
1. The system prompt is *always* in context. Adding 200 words there costs 200 tokens per LLM turn for every query, even queries that do not need that guidance.
2. Long system prompts cause **instruction skimming**: agents follow the structural rules at the top and ignore negative constraints buried below (typical onset around 3000 tokens).
3. System-prompt edits are global and cannot be conditionally activated.

A well-scoped **optional skill** sidesteps all three:
- Only its `name` + `description` (one line) sits in the system prompt.
- The body is loaded on demand via `load_skill(<name>)` and appears as a tool result.
- It is conditionally activated when the agent decides it needs that strategy.

**Default to skills, not prompt edits** when the behaviour is conditional on query type. Repeated optimization runs that only edit `prompts/<agent>/system.md` plateau quickly because every fix is global and cumulative.

# Three skill categories — when to use which

| Category | Where listed in `config.yaml` | Where the body appears | When to use |
|---|---|---|---|
| `system_skills` | `system_skills:` list | Inlined verbatim into the system prompt as `### Protocol N: <name>` | Foundational behaviour the agent must always follow (workspace conventions, TODO discipline, phase rules). Keep ≤3. |
| `preload_skills` | `preload_skills:` list | Injected into the initial user message AND re-injected after each `phase_done_and_summarize` | Behaviours that should persist across phases but do not justify burning system-prompt tokens. Survives phase compression. |
| `optional_skills` | `optional_skills:` list | Loaded on demand via `load_skill(<name>)`; appears as a tool result | Specialised strategies for specific failure modes or query subtypes. Add freely — they are nearly free until used. |

A skill can appear in both `preload_skills` and `optional_skills`. The latter allows the agent to manually re-load the body if the conversation has drifted.

# Skill file structure

Every skill is a **folder** under `mas_config/skills/<agent_id>/`:
```
mas_config/skills/<agent_id>/
  <skill-name>/
    SKILL.md          ← required; the only file the loader reads
    extra_content/    ← optional reference material the SKILL.md body may cite
```
The folder name **must exactly match** the `name` frontmatter field. Mismatched names are a fatal load error.

# SKILL.md format
```
---
name: <skill-name>            # MUST match folder name
description: <one-line desc>  # appears in system prompt for optional skills
---

<body — instructions the agent will follow>
```
**Body formatting depends on category:**
- **System skills**: body is inlined under a `### Protocol N: <name>` heading in the system prompt. Therefore the body MUST NOT contain markdown headings (`#`, `##`, `###`). Use `**bold text**` for sub-sections. Headings would corrupt the system-prompt heading hierarchy.
- **Preload and optional skills**: body is delivered as tool output, so all markdown features (headings, tables, code blocks) are allowed.

The `description` field of an optional skill is the **only** text the agent sees by default. It must clearly answer: *"what problem does this skill solve and when should I load it?"* Vague descriptions mean the agent never loads the skill — verify by checking `agent_stats.skill_calls` in batch reports.

# Skill body design heuristics

A good skill body is structured as:
1. **Trigger conditions** — bullet list of observable signals telling the agent when to load this skill. Be concrete (numeric thresholds, specific situations), not vague ("when stuck").
2. **The workflow** — numbered steps. Each step should be one decision or one action. Avoid meta-commentary.
3. **Anti-patterns** — explicit "do not do X" items addressing the failure mode the skill is correcting.
4. **Output / verification contract** — what the agent should produce or check before exiting the workflow.

Bodies that are pure prose are skimmed. Bodies that are checklists are followed.

# When to create a new skill

Create a new skill when **all** of the following are true:
1. A failure pattern (or successful pattern worth preserving) is **systemic** (≥2 queries, or 1 query plus alignment with a prior `applied`/`confirmed` hypothesis).
2. The pattern is amenable to a **specific corrective workflow** that an LLM can follow when shown to it (rules, checklists, decision trees).
3. The same workflow does NOT need to be active on every query (otherwise make it a system or preload skill).

Do **not** create a skill for:
- A one-off failure on a single query.
- Vague advice ("be more careful with sources") — agents ignore it.
- Behaviour already covered by an existing skill (extend the existing skill).

# Worked example — converting a pattern into a skill (domain-neutral)

Suppose `trajectory-analysis` shows:
- Across 3 failing queries, the agent calls tool T₁ 30+ times per query, never calls tool T₂ (which has complementary capabilities), and the audit log shows the agent abandoning T₁'s output without acting on it.
- The successful query in the same batch alternated T₁ and T₂ in a 3:1 ratio.

The pattern is **mono-tool fixation on T₁** (a universal abnormal pattern).

## Step 1 — name and scope the skill
`name: t1-t2-pivot`

Description: "When repeated calls to T₁ stop producing useful new information, pivot to T₂ to consume T₁'s most promising output in depth before continuing."

## Step 2 — write the body (template)
```
---
name: t1-t2-pivot
description: When repeated calls to T₁ stop producing useful new information, pivot to T₂ to consume T₁'s most promising output in depth before continuing.
---

# When to load this skill
Load when ALL of these are true:
- You have called T₁ ≥10 times for the current query.
- The most recent 3 T₁ results contain candidates you have not yet examined with T₂.
- You cannot answer the question with the information currently in your context.

# The workflow
1. Pick the top N candidates (N ≤ 3) from your most recent T₁ results.
2. For each candidate, call T₂ to extract the specific attribute(s) the task requires.
3. If T₂ confirms an answer, cite the T₂ output verbatim in your final answer.
4. If T₂ returns "not found" on all candidates, the candidates were wrong — refine your T₁ strategy; do NOT pick more candidates from the same T₁ results.

# Anti-patterns this skill prevents
- Calling T₁ 40+ times without ever calling T₂.
- Calling T₂ on too many candidates at once and overflowing context.
- Asking T₂ vague questions instead of targeted ones.

# Verification contract
Before `task_done`, confirm that your final answer cites the specific T₂ output that supports it. If you cannot, do not finalize.
```

## Step 3 — register and verify
1. `write_file mas_config/skills/<agent_id>/t1-t2-pivot/SKILL.md` with the content above.
2. `file_edit mas_config/config.yaml` to add `"t1-t2-pivot"` to the agent's `optional_skills` list.
3. Confirm `load_skill` is in `available_tools` (required for any optional skill to work).
4. `finalize_optimization` validates file existence + YAML.
5. Record a hypothesis: ID Hxx, Category negative, Observation "mono-tool fixation on T₁ in 3 queries", Action "create skill `t1-t2-pivot`", Expected outcome "T₁ count drops, T₂ count rises, accuracy on residual class improves".

## Step 4 — verify uptake in the next batch
Check `agent_stats.skill_calls` for `t1-t2-pivot` count > 0 and the predicted tool-distribution change. If the skill was never loaded, the description was not specific enough — refine it.

# Editing an existing skill is often the right move

- Same topic but missing a case → extend the existing skill.
- Conflicting guidance with an existing skill → reconcile in one place; do not create a competing skill.
- Skill is too long / agent skips parts of it → **split** into two narrower skills with sharper trigger conditions.

When you split or rewrite, mark the old hypothesis as `superseded` and link to the replacement.

# Promoting / demoting between categories

| Movement | Trigger |
|---|---|
| optional → preload | Skill is loaded on most queries; preload saves the load_skill call. |
| optional → system | Skill is loaded on every query AND fits in the system-prompt budget AND does not require headings. |
| system → optional | Skill is being skimmed/ignored AND is not actually needed every query. |
| preload → optional | Skill is rarely relevant. |

Each movement is a hypothesis. Record it.

# Pre-`finalize_optimization` checklist

1. SKILL.md frontmatter `name` matches folder name exactly.
2. If system skill: zero markdown headings in body.
3. If preload or optional: heading-free is preferred but not required.
4. If optional or preload: `load_skill` is in the agent's `available_tools`.
5. Description is specific enough that the agent can decide when to load.
6. The skill addresses a pattern documented in the hypothesis ledger.
