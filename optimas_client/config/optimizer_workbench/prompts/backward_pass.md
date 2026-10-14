## MAS Optimization Task — Backward Pass

{progress}

### Validation signal
{validation_metrics}

### Previous optimization
{prev_summary}

### Training batch diagnostics
{training_batch_diagnostics}

{task_specific_evaluation_guide}

### Artifact paths
{artifact_navigation}

---

## Your task this step

Run one backward pass over the batch above, following the workflow, action space,
escalation principles and key constraints already given in your system prompt:

1. **Review history** — `optimizer/optimization_history.json` and
   `optimizer/hypothesis_ledger.md`. Re-evaluate every `applied`/`confirmed`
   hypothesis against this batch.
2. **Gather evidence** — read `tool_call/tool_calls.jsonl` for each failing query you
   triage before asking `text_qa` anything. Prioritise **resampled** failures: they are
   direct feedback on your previous intervention.
3. **Formulate / update hypotheses** — one hypothesis per intended change.
4. **Choose the lowest-level intervention** whose scope matches the diagnosed root
   cause; escalate only on retrospective or structural evidence.
5. **Apply changes** under `mas_config/`, recording every changed file in the ledger's
   **Files changed** sub-field.
6. **Finalize** — call `finalize_optimization`, fix any reported errors, then `task_done`.

Remember: exactly one change per hypothesis, no change without evidence, and never
embed query-specific facts or gold-answer entities in any MAS file.
