---
name: optimizer-toolkits
description: Universal tool-design theory for MAS agents — how `available_tools` shapes behaviour, design principles, the framework tool catalog with deep per-tool references (strengths, limitations, synergies), and the decision checklist for tool-set changes.
---

# How `available_tools` shapes agent behaviour

Every entry in an agent's `available_tools` list adds tool-schema tokens to **every** LLM call and adds one more option the agent must choose between. Tool selection is itself a strong inductive bias on the agent's behaviour:
- An agent given only one research tool will solve everything with that tool.
- An agent given a diverse tool set will use the dominant one in its training prior unless the system prompt or skills explicitly point to alternatives.
- Removing a tool the agent overuses can force a behaviour change with **zero** prompt-text changes.

**Tool-set changes are often higher-leverage than prompt edits**, and they are completely missed if the only edit lever you ever pull is the system prompt.

# Two orthogonal tool dimensions

When designing an agent's tool set, think in two dimensions:
1. **Capability surface** — what classes of action the agent can perform. Examples: file I/O, code execution, web access, long-document reading, structured extraction, image/video analysis, delegation.
2. **Granularity** — within a class, the level at which actions are exposed. A single broad tool that does many things vs. several narrow tools each doing one thing. Narrow tools are easier for the agent to choose correctly but add schema cost; broad tools are cheaper but more error-prone.

Tool design = pick the smallest capability surface that covers the task domain, then pick the granularity that the underlying LLM can reliably select between.

# Framework tool catalog — quick reference

The tables below list every tool this deployment's MCP server actually registers. **Never add a tool to `available_tools` that does not appear here** — unknown names are rejected at MAS validation. For **deep per-tool documentation** (all parameters, implementation details, hard limits, synergies, overlap with other tools, and common failure patterns), load the corresponding reference file from `extra_content/`:

## File operations (universal)
| Tool | One-line summary | Reference |
|---|---|---|
| `read_file` | Read text/rich-format files with optional line range. 40k-char output cap. | `extra_content/read_file.md` |
| `write_file` | Create / overwrite / append files. Auto-creates parent dirs. | `extra_content/write_file.md` |
| `file_edit` | Line-based content replacement with context matching. Also: `patch_file`, `replace_file_block`. | `extra_content/file_edit.md` |
| `grep_file_content` | Pattern search (literal or regex via ripgrep) with optional replacement. 30s timeout. | `extra_content/grep_file_content.md` |
| `list_file_tree` | Directory tree listing with depth control. | `extra_content/list_file_tree.md` |
| `delete_file` | Delete file or directory (recursive). No undo. | `extra_content/delete_file.md` |
| `move_file` | Move or copy a single file. | `extra_content/move_file.md` |

## Content analysis — LLM-powered (universal)
| Tool | One-line summary | Reference |
|---|---|---|
| `text_qa` | Sliding-window LLM Q&A over text/rich files. Expensive but handles arbitrarily large files. | `extra_content/text_qa.md` |
| `text_extract` | Structured multi-file JSON extraction with halo chunking. Batch triage. | `extra_content/text_extract.md` |
| `image_qa` | Single-round VLM Q&A over images (JPEG/PNG/GIF/WebP). Rejects non-images. | `extra_content/image_qa.md` |

## Workflow control
| Tool | One-line summary | Reference |
|---|---|---|
| `load_skill` | Load an optional skill body into conversation. | `extra_content/workflow_tools.md` |
| `task_done` | End the agent's task; `task_summary` is returned to caller. | `extra_content/workflow_tools.md` |
| `phase_done_and_summarize` | End current phase, auto-compress, start new phase. | `extra_content/workflow_tools.md` |
| `assign_task_to_agent` | Delegate to a named sub-agent. Blocks until sub-agent completes. | `extra_content/workflow_tools.md` |
| `finalize_optimization` | (Optimizer-only) Validates MAS structure + YAML before `task_done`. | `extra_content/workflow_tools.md` |

# Key tool synergies and overlaps

## The "discover → retrieve → understand" pipeline
Most information-processing tasks follow this three-stage pipeline, regardless of domain:

| Stage | Tools |
|---|---|
| **Discover** | `list_file_tree`, `grep_file_content` |
| **Retrieve** | `read_file` (with line ranges) |
| **Understand** | `read_file` (small), `text_qa` (large), `image_qa` (image) |

Agents that skip stages produce lower-quality results. Agents that repeat early stages without advancing show mono-tool fixation.

## Critical overlaps to manage in skills
| Overlap | Risk | Recommendation |
|---|---|---|
| `read_file` vs `text_qa` | Agent defaults to `text_qa` for small files (wasteful) or `read_file` for large files (truncation). | Skill: "use `read_file` for files < 500 lines or known line range; `text_qa` when answer location unknown in large file." |
| `file_edit` vs `write_file` | Agent uses `write_file` to "edit" → overwrites and destroys unread content. | Skill: "never use `write_file` to modify existing files; use `file_edit` or `patch_file`." |
| `grep_file_content` (replace) vs `file_edit` | Agent grep-replaces with broad pattern → unintended changes. | Skill: "use grep-replace only for batch substitutions with `path` restricted." |
| `text_qa` vs `text_extract` | Agent calls `text_qa` on 5 files sequentially (slow) or `text_extract` for one small file (overkill). | Skill: "use `text_extract` for batch triage across 3+ files; `text_qa` for deep single-file investigation." |

# Tool-set design heuristics (universal)

1. **Start small, expand on evidence.** A new agent should begin with the smallest tool set that plausibly covers the task. Add a tool when the trajectory shows a recurring need.
2. **Pair every addition with a removal when possible.** Long `available_tools` lists bloat every LLM call's schema and increase decision noise.
3. **Capability gaps cause silent failures.** If the agent never solves problems of class X, check whether it has any tool whose capability surface covers X.
4. **Capability overlaps cause inconsistency.** If two tools can do the same thing, make the agent's preferred order explicit in a skill.
5. **A tool the agent never calls is dead weight.** Check `agent_stats.tool_distribution` in the batch report. Tools with 0 calls are removal candidates — but verify they were never *needed* first.
6. **A tool that always fails is not a MAS issue.** Repeated failures in `tool_calls.jsonl` = infrastructure breakage. Record and move on.
7. **Granting a tool without guidance often changes nothing.** Pair every tool addition with either a skill update or a one-line system-prompt trigger condition.

# When to change `available_tools` — decision checklist

Before any change, your hypothesis must concretely answer:
1. **Evidence in `tool_calls.jsonl`**: how many times was each existing tool called? Success rate? Which results were ignored?
2. **Predicted behaviour change**: expected change in tool distribution after the next batch.
3. **Pairing**: is the change accompanied by a skill or prompt update?
4. **What you remove**: every addition should ideally be matched by a removal; justify the net schema-cost increase if not.

If you cannot answer (1) and (2) concretely, you do not have enough evidence.

# When to remove a tool

- Zero calls across multiple steps AND no expected need → remove.
- Repeated failures with no upstream fix → keep but record infra hypothesis.
- Heavy use but consistently low marginal value → keep but improve result-handling via a skill before considering removal.
