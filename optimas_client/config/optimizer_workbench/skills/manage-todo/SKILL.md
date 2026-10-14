---
name: manage-todo
description: Maintain a structured TODO file as the optimizer's persistent memory across phases.
---

Keep **one** TODO file at `optimizer/todo.md`. This file is your **persistent memory** — when `phase_done_and_summarize` compresses your conversation context, the TODO file on disk survives intact. Write into it early and often.

**Structure:** organize the TODO file into sections:

```
**Observations**
- [finding from trajectory analysis, failure patterns, key metrics]
- [what the validation signal implies]

**Plan**
- [ ] [planned change with brief rationale]
- [ ] [planned change with brief rationale]

**Changes Made**
- [x] [what was changed, which file, why]
- [x] [what was changed, which file, why]

**Dropped**
- [~] [task description] — reason: [why it was dropped or deferred]

**Notes**
- [hypotheses, ideas for future steps, things to revisit]
```

**Status markers** — at the very start of the line, exactly once:
- `[ ]` not started
- `[r]` in progress
- `[x]` done
- `[~]` dropped or deferred (always include the reason)

**Active maintenance — the TODO is a living document, not a write-once log:**

1. **Add** new items as you discover new findings or plan new changes.
2. **Modify** existing items when your understanding changes (e.g., refine a plan after deeper analysis).
3. **Drop** items that are no longer relevant: preserve their text, mark them `[~]`, and include `reason: ...` on the same line. Preserve the plan's audit trail rather than erasing entries.
4. **Review & mark completion** — after finishing each change, immediately mark it `[x]` and move it to **Changes Made** with the file path and reason. Do not leave completed work unmarked.

**Mandatory review checkpoints:**
- **Before `phase_done_and_summarize`**: review every item in **Plan** — mark completed items `[x]`, update in-progress items `[r]`, drop irrelevant items `[~]`.
- **Before `finalize_optimization`**: create the TODO file and resolve every checklist item as `[x]` (done) or `[~]` (dropped with `reason: ...`). This check is mandatory; missing, empty, or unresolved plans block delivery. Record future work as a justified dropped item in this step.

**When to update:** update the TODO file at every decision point — after analyzing trajectories, before making a change, after completing a change, and before calling `phase_done_and_summarize` or `finalize_optimization`. Do not batch all updates to the end; write as you go so nothing is lost if context is compressed.
