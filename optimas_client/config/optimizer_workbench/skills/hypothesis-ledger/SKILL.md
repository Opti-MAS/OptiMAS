---
name: hypothesis-ledger
description: Maintain a persistent hypothesis ledger to drive principled, evidence-based MAS optimization across steps.
---

**File**: `optimizer/hypothesis_ledger.md` — your long-term optimization memory. This file persists across ALL steps and epochs (never archived into working logs). Every optimization decision must trace back to a hypothesis in this file.

**Core rule: no change without a hypothesis.** Before modifying any MAS config file, you must have a recorded hypothesis that motivates the change. Undirected changes cannot be evaluated or reverted intelligently.

**Hypothesis structure** — each entry must include:
- **ID**: Sequential (H1, H2, …). Never reuse IDs.
- **Step**: Step number when created.
- **Category**: `positive` (effective behavior to preserve/reinforce) or `negative` (failure pattern to fix).
- **Observation**: Specific behavior seen in trajectories (with query IDs, tool counts, or quotes).
- **Root cause / Mechanism**: Why this behavior occurs.
- **Action level**: L1 / L2 / L3 / L4 (see leveled action space in system prompt). For L3+, note why lower levels are insufficient.
- **Action**: Concrete MAS config change description, or "monitor" for positive patterns.
  - **Files changed**: list of MAS files modified (e.g., `skills/worker/my-skill/SKILL.md`, `prompts/worker/system.md`).
- **Expected outcome**: Measurable prediction (e.g., "reduce avg calls of tool T from 12 to <8", "raise accuracy on query subclass C by ≥5%").
- **Status**: see lifecycle below.
- **Evidence**: Updated each step with relevant observations.

**Status lifecycle:**
- `pending` — hypothesis formulated but no MAS change applied yet. Use when further analysis is needed before committing a change.
- `applied` — MAS change implemented based on this hypothesis, **awaiting empirical validation**. Most new hypotheses with changes start here. The change looks reasonable but its effectiveness has not been proven by subsequent step results.
- `confirmed` — **evidence-supported**. Validated by empirical results or rigorous logical analysis. Prioritize preserving these insights, but review periodically — a hypothesis confirmed on one batch may prove ineffective on later batches. If counter-evidence emerges, transition to `superseded` or `refuted`. Criteria:
  - (a) Error pattern directly observable in trajectory evidence AND the fix logically addresses the root cause. May be set at creation.
  - (b) An `applied` hypothesis validated by ≥1 subsequent step (target pattern improved or disappeared).
  - Prefer `applied` over `confirmed` when effectiveness is not yet demonstrated.
- `unverified` — after multiple steps, no relevant scenarios appeared to test this hypothesis.
- `refuted` — **training trajectory evidence** confirms the hypothesis was wrong or the change caused concrete negative effects in agent behaviour (e.g., the change introduced confusion, wasted turns, led to wrong fixes). Val accuracy decline (on the fixed validation set) is a meaningful warning that should trigger investigation, but **accuracy metrics alone — whether val or train — are NOT sufficient evidence for refutation**. Note: you do NOT have access to validation trajectories — only training batch trajectories are available for analysis. Aggregate train accuracy is unreliable because each batch mixes resampled queries (from previous failures) and freshly sampled queries (new problems). **However, resampled queries that still fail are strong direct evidence** about your last intervention's effectiveness. To refute, you must demonstrate via **training trajectory analysis** that the targeted behaviour actually worsened or that the change produced a new failure mode. Revert and create a replacement.
- `superseded` — replaced by a refined hypothesis (note the new ID).

**Transition rules:**
```
pending → applied    (change implemented)
pending → confirmed  (only if evidence is irrefutable)
applied → confirmed  (subsequent step evidence validates effectiveness)
applied → unverified (2+ steps with no relevant cases)
applied → refuted    (trajectory analysis confirms change caused concrete negative effects — accuracy decline is a trigger for investigation, not standalone evidence)
applied → superseded (replaced by a refined hypothesis)
```

**Mandatory workflow at EACH step:**
1. **Read** `optimizer/hypothesis_ledger.md` at the start.
2. **Review** each active hypothesis against this step's batch results and trajectories:
   - `applied` → evaluate using **both** accuracy signals **and** training trajectory analysis. You can only analyze **training batch trajectories** — validation trajectories are not available, only the val accuracy metric is provided as an overfitting guard. Val accuracy decline (fixed val set) is a meaningful warning that should prompt deeper investigation. Aggregate train accuracy is a mixed signal — each batch contains **resampled queries** (from previous failures, marked `is_resampled`) and **freshly sampled queries** (new problems). Aggregate train Δ is unreliable, but **resampled queries that still fail are strong direct feedback** on your last intervention. Check whether the specific behaviour the hypothesis targeted actually changed in the training trajectories (especially resampled failures). Promote to `confirmed` only with clear trajectory-level support. Demote to `refuted` only if training trajectory analysis confirms the change caused concrete negative effects — accuracy decline alone (even val Δ) is not sufficient because it could stem from LLM randomness or unrelated factors. Keep `applied` if evidence is ambiguous. Transition to `unverified` if 2+ steps passed with no relevant cases.
   - `confirmed` → re-evaluate against current evidence; demote to `refuted`/`superseded` if contradicted, or to `unverified` if conditions shifted.
   - `pending` → decide: implement (→ `applied`) or defer.
3. **Update statuses** with specific evidence notes.
4. **Resolve conflicts** — if two hypotheses contradict, supersede the weaker one.
5. **Analyze trajectories** — extract new positive and negative patterns (see trajectory-analysis skill).
6. **Formulate new hypotheses** for any new patterns before making changes:
   - Set `confirmed` at creation **only** when the error is directly observable AND the fix is a logical certainty (rare).
   - Set `applied` for all other hypotheses where you implement a change.
   - Set `pending` when you identify a pattern but are not ready to change.
7. **Make MAS changes** driven by hypotheses. Record every file you changed in the hypothesis ledger's **Files changed** sub-field — this is the ONLY place where hypothesis-to-change traceability is kept.
8. **Record** all new and updated hypotheses before `finalize_optimization`.

**Ledger file format:**
```
# Hypothesis Ledger

## Active Hypotheses

### H<N> [status] (Step <created>)
- **Category**: positive / negative
- **Observation**: <specific behavior with evidence>
- **Root cause**: <why this happens>
- **Action [L<level>]**: <MAS change description> | monitor
  - **Files changed**: <list of MAS files modified, e.g. `skills/worker/my-skill/SKILL.md`, `prompts/worker/system.md`>
- **Escalation reasoning** (for L3+): <why lower levels are insufficient>
- **Expected outcome**: <measurable prediction>
- **Evidence**:
  - Step <X>: <observation>

## Archived Hypotheses
(Hypotheses confirmed over 3+ steps or permanently refuted — moved here to keep Active section focused)
```

**Before `phase_done_and_summarize`:** write all pending hypothesis updates to the ledger file. The file on disk survives; your conversation memory does not.

**Before `finalize_optimization`:** final review — every new hypothesis from this step must be recorded; every MAS change must be traceable to a hypothesis via the ledger's **Files changed** sub-field.

**CRITICAL: hypothesis IDs must NEVER appear in MAS text.** Do NOT embed identifiers like `H1`, `H5`, `(ref: H12)`, or any hypothesis-ledger notation in any file under `mas_config/` — not in prompts, skills, or config YAML. Hypothesis traceability is maintained exclusively in the ledger's structured **Action → Files changed** field. MAS agents cannot interpret hypothesis IDs; their presence in prompts or skills degrades agent performance.