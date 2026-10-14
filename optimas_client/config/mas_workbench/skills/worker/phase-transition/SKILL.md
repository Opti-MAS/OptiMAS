---
name: phase-transition
description: Decide when to call phase_done_and_summarize versus task_done.
---

- Call `phase_done_and_summarize` when there is remaining work or the context is getting long. This compresses history and continues execution.
- Call `task_done` only when the task is fully complete.

**Writing the `progress_report`:** The system will auto-compress your conversation to carry forward a detailed work log. Your progress report should complement this by highlighting:
1. **Findings so far**: key facts discovered, sources verified, hypotheses confirmed or eliminated.
2. **Current best answer candidate** (if any) and which constraints are still unverified.
3. **What did NOT work**: search queries that returned no useful results, leads that were dead ends.
4. **Immediate next step**: what should be done first when the next phase starts.
