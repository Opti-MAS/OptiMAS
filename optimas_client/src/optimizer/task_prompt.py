"""
Backward-pass task prompt construction for the optimizer.

The template ``backward_pass.md`` uses ``str.format`` placeholders:
``{validation_metrics}``, ``{prev_summary}``, ``{training_batch_diagnostics}``,
``{artifact_navigation}``.  Python fills each from ``optimization_inputs``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# §1  Formatting utilities
# ═══════════════════════════════════════════════════════════════════════════


def _markdown_table(rows: List[tuple]) -> str:
    """Two-column Metric | Value table."""
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _fmt_dist(dist: Optional[Dict[str, Any]], limit: int = 8) -> str:
    """Format a {name: avg_count} distribution dict as a compact string."""
    if not dist:
        return "(none)"
    items = sorted(dist.items(), key=lambda x: -x[1])
    parts = [f"{n}: {v}" for n, v in items[:limit]]
    if len(items) > limit:
        parts.append(f"... +{len(items) - limit} more")
    return ", ".join(parts)


def _fmt_num(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}" if v != int(v) else str(int(v))
    return str(v)


def _trunc(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _fmt_query_agent_stats(agent_stats: Any) -> str:
    """One-line compact summary of per-query agent work for the results table.

    Example output: ``worker: 17r/1p/17t``  (rounds / phases / tool_calls).
    For multi-agent: ``worker: 17r/1p/17t; researcher: 8r/2p/6t``.
    """
    if not isinstance(agent_stats, dict) or not agent_stats:
        return "N/A"
    parts = []
    for name, s in sorted(agent_stats.items()):
        if not isinstance(s, dict):
            continue
        r = s.get("rounds", "?")
        p = s.get("phases", "?")
        t = s.get("tool_calls", "?")
        parts.append(f"{name}: {r}r/{p}p/{t}t")
    return "; ".join(parts) if parts else "N/A"


# ═══════════════════════════════════════════════════════════════════════════
# §2  Data extraction helpers
# ═══════════════════════════════════════════════════════════════════════════


def _train_stats(train_perf: Dict[str, Any]) -> Dict[str, Any]:
    """Accuracy counters from ``stats`` or ``batch_aggregates`` (excluding efficiency)."""
    s = train_perf.get("stats") or {}
    if s:
        return s
    br = train_perf.get("batch_report")
    if not isinstance(br, dict):
        return {}
    ba = br.get("batch_aggregates") or {}
    return {k: v for k, v in ba.items() if k != "efficiency"}


def _wall_stats_from_queries(queries: List[Any]) -> Tuple[Optional[float], Optional[float]]:
    ws: List[float] = []
    for q in queries or []:
        if not isinstance(q, dict):
            continue
        wt = q.get("wall_time_sec")
        if wt is None:
            wt = (q.get("run") or {}).get("wall_time_sec")
        if isinstance(wt, (int, float)):
            ws.append(float(wt))
    if not ws:
        return (None, None)
    return (round(sum(ws) / len(ws), 2), round(max(ws), 2))


def _tool_union_from_per_agent(per: Dict[str, Any]) -> str:
    names: set = set()
    for m in (per or {}).values():
        if not isinstance(m, dict):
            continue
        for n in m.get("tool_names_observed") or []:
            names.add(str(n))
    return ", ".join(sorted(names)) if names else "(none)"


# ═══════════════════════════════════════════════════════════════════════════
# §3  Metric block builders
# ═══════════════════════════════════════════════════════════════════════════


def build_validation_metrics(optimization_inputs: Dict[str, Any]) -> str:
    val_signal = optimization_inputs.get("validation_signal", {}) or {}
    current_acc = float(val_signal.get("current_acc", 0.0))
    acc_diff = float(val_signal.get("acc_diff", 0.0))

    rows = [
        ("**Validation accuracy**", f"**{current_acc:.4f}**"),
        ("**Accuracy Δ from previous step**", f"**{acc_diff:+.4f}**"),
    ]

    return "\n\n".join([
        _markdown_table(rows),
        ("> A negative Δ is a generalization warning. Inspect training trajectories "
         "before attributing a regression or refuting a hypothesis."),
    ])


def build_training_batch_diagnostics(optimization_inputs: Dict[str, Any]) -> str:
    """Unified batch diagnostics: overview + per-query results + per-agent efficiency."""
    train_perf = optimization_inputs.get("train_performance", {}) or {}
    stats = _train_stats(train_perf)
    br = train_perf.get("batch_report")
    queries: List[Dict[str, Any]] = []
    ba: Dict[str, Any] = {}
    per: Dict[str, Any] = {}

    if isinstance(br, dict):
        queries = br.get("queries") or []
        ba = br.get("batch_aggregates") or {}
        per = ba.get("per_agent") or {}

    n = len(queries) or len(train_perf.get("batch") or [])

    # -- wall time --
    avg_w = ba.get("avg_mas_wall_time_sec")
    max_w = ba.get("max_mas_wall_time_sec")
    if avg_w is None or max_w is None:
        qavg, qmax = _wall_stats_from_queries(queries)
        avg_w = avg_w if avg_w is not None else qavg
        max_w = max_w if max_w is not None else qmax

    # -- tool union --
    u = ba.get("unique_tool_types_union")
    u_str = ", ".join(u) if isinstance(u, list) and u else _tool_union_from_per_agent(per)

    parts: List[str] = []

    is_workbench = stats.get("backend") == "workbench"

    # ── Section 1: Batch overview ──
    if is_workbench:
        overview_rows = [
            ("Batch size", str(n)),
            ("State accuracy", f"{stats.get('accuracy', 'N/A')} "
             f"({stats.get('success_count', '?')} correct / {stats.get('failure_count', '?')} fail)"),
            ("Unwanted side effects / no-action / agent errors",
             f"{stats.get('unwanted_side_effects_count', 0)} / {stats.get('no_action_count', 0)} / {stats.get('agent_error_count', 0)}"),
            ("Avg / max wall time (s)", f"{_fmt_num(avg_w)} / {_fmt_num(max_w)}"),
            ("Tools used (all agents)", u_str),
        ]
    else:
        overview_rows = [
            ("Batch size", str(n)),
            ("Accuracy", f"{stats.get('accuracy', 'N/A')} "
             f"({stats.get('success_count', '?')} correct / {stats.get('wrong_answer_count', '?')} wrong / {stats.get('agent_error_count', 0)} agent errors)"),
            ("Avg / max wall time (s)", f"{_fmt_num(avg_w)} / {_fmt_num(max_w)}"),
            ("Tools used (all agents)", u_str),
        ]
    parts.append(_markdown_table(overview_rows))

    # ── Section 2: Per-query results ──
    if queries:
        if is_workbench:
            has_resampled = any(q.get("is_resampled") for q in queries)
            _MAX_REASON_CHARS = 200
            if has_resampled:
                header = [
                    "",
                    "**Per-query results:** (★ = resampled from previous step's failures):",
                    "",
                    "| query_id | ★ | correct | exact | side_fx | #pred | #gold | wall_time (s) | agent behavior | reason |",
                    "|----------|---|---------|-------|---------|-------|-------|---------------|----------------|--------|",
                ]
            else:
                header = [
                    "",
                    "**Per-query results:**",
                    "",
                    "| query_id | correct | exact | side_fx | #pred | #gold | wall_time (s) | agent behavior | reason |",
                    "|----------|---------|-------|---------|-------|-------|---------------|----------------|--------|",
                ]
            rows: List[str] = []
            for q in queries:
                qid = q.get("query_id", "?")
                correct = "pass" if q.get("correct") else "FAIL"
                exact = "yes" if q.get("exact_match") else "no"
                side_fx = "YES" if q.get("unwanted_side_effects") else "no"
                n_pred = q.get("predicted_actions_count", "?")
                n_gold = q.get("ground_truth_actions_count", "?")
                wt = q.get("wall_time_sec")
                wt_s = f"{wt:.1f}" if isinstance(wt, (int, float)) else "N/A"
                reason = str(q.get("evaluation_reason") or "")
                reason = _trunc(reason.replace("\n", " ").strip(), _MAX_REASON_CHARS)
                behavior = _fmt_query_agent_stats(q.get("agent_stats"))
                if has_resampled:
                    resample_mark = "★" if q.get("is_resampled") else ""
                    rows.append(f"| {qid} | {resample_mark} | {correct} | {exact} | {side_fx} | {n_pred} | {n_gold} | {wt_s} | {behavior} | {reason} |")
                else:
                    rows.append(f"| {qid} | {correct} | {exact} | {side_fx} | {n_pred} | {n_gold} | {wt_s} | {behavior} | {reason} |")
            parts.append("\n".join(header + rows))
        else:
            _MAX_GRADER_CHARS = 100
            header = [
                "",
                "**Per-query results:**",
                "",
                "| query_id | score | wall_time (s) | agent behavior | grader summary |",
                "|----------|-------|---------------|----------------|----------------|",
            ]
            rows = []
            for q in queries:
                qid = q.get("query_id", "?")
                score = q.get("score")
                if score is None:
                    score = (q.get("evaluation") or {}).get("score", "?")
                wt = q.get("wall_time_sec")
                if wt is None:
                    wt = (q.get("run") or {}).get("wall_time_sec")
                wt_s = f"{wt:.1f}" if isinstance(wt, (int, float)) else "N/A"
                grader = str(q.get("grader_response") or q.get("reasoning")
                             or (q.get("evaluation") or {}).get("grader_response") or "")
                grader = _trunc(grader.replace("\n", " ").strip(), _MAX_GRADER_CHARS)
                behavior = _fmt_query_agent_stats(q.get("agent_stats"))
                rows.append(f"| {qid} | {score} | {wt_s} | {behavior} | {grader} |")
            parts.append("\n".join(header + rows))

    # ── Section 3: Per-agent efficiency ──
    if isinstance(per, dict) and per:
        sorted_aids = sorted(per.keys(), key=lambda a: (0 if (per[a] or {}).get("is_entry_agent") else 1, a))

        for aid in sorted_aids:
            m = per[aid] or {}
            if not isinstance(m, dict):
                continue
            role = "entry" if m.get("is_entry_agent") else "sub-agent"

            agent_rows: List[tuple] = [
                (f"**`{aid}`** ({role})", ""),
            ]
            if not m.get("is_entry_agent"):
                agent_rows.append(("Invocations per query (avg/max)",
                                   f"{m.get('avg_invocations_per_query', 'N/A')} / {m.get('max_invocations_per_query', 'N/A')}"))

            agent_rows.extend([
                ("ReAct steps (avg/max)", f"{m.get('avg_assistant_rounds', 'N/A')} / {m.get('max_assistant_rounds', 'N/A')}"),
                ("Phases — context resets (avg/max)", f"{m.get('avg_phases', 'N/A')} / {m.get('max_phases', 'N/A')}"),
                ("Tool calls (avg/max)", f"{m.get('avg_tool_invocations', 'N/A')} / {m.get('max_tool_invocations', 'N/A')}"),
                ("Failed tool calls (avg/max)", f"{m.get('avg_failed_tool_calls', 'N/A')} / {m.get('max_failed_tool_calls', 'N/A')}"),
                ("Tool distribution", _fmt_dist(m.get("tool_call_distribution"))),
            ])
            skill_dist = m.get("skill_call_distribution")
            if skill_dist:
                agent_rows.append(("Skill distribution", _fmt_dist(skill_dist)))
            deleg_dist = m.get("delegation_distribution")
            if deleg_dist:
                agent_rows.append(("Delegation distribution", _fmt_dist(deleg_dist)))

            parts.append(_markdown_table(agent_rows))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# §4  Template context builder
# ═══════════════════════════════════════════════════════════════════════════


def build_progress_line(optimization_inputs: Dict[str, Any]) -> str:
    """Single-line progress indicator for the optimizer."""
    p = optimization_inputs.get("progress") or {}
    if not p:
        return ""
    cur_epoch = p.get("current_epoch", "?")
    total_epochs = p.get("total_epochs", "?")
    cur_step = p.get("current_step", "?")
    steps_per_epoch = p.get("steps_per_epoch", "?")
    total_steps = p.get("total_steps", "?")
    return (
        f"**Progress: Epoch {cur_epoch}/{total_epochs}, "
        f"Step {cur_step}/{total_steps} "
        f"({steps_per_epoch} steps per epoch)**"
    )


def _build_default_context(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "progress": build_progress_line(inputs),
        "validation_metrics": build_validation_metrics(inputs),
        "prev_summary": inputs.get("previous_optimization", "None"),
        "training_batch_diagnostics": build_training_batch_diagnostics(inputs),
        "artifact_navigation": build_artifact_navigation_block(inputs),
        "task_specific_evaluation_guide": build_task_specific_evaluation_guide(inputs),
    }


# ═══════════════════════════════════════════════════════════════════════════
# §5  Public API
# ═══════════════════════════════════════════════════════════════════════════


def build_backward_pass_task_prompt(
    template: str,
    optimization_inputs: Dict[str, Any],
) -> str:
    """Fill *template* with context derived from *optimization_inputs*.

    Uses ``str.format_map`` with a fallback dict so that unknown
    placeholders (e.g. literal ``{task}`` in documentation text) are
    left as-is instead of raising ``KeyError``.
    """
    ctx = _build_default_context(optimization_inputs)

    class _SafeDict(dict):
        """Return the placeholder itself for missing keys."""
        def __missing__(self, key: str) -> str:
            logger.warning(
                "backward_pass template contains unknown placeholder '{%s}'; "
                "leaving it as-is. Check the template for unescaped braces.",
                key,
            )
            return "{" + key + "}"

    return template.format_map(_SafeDict(ctx))


def build_artifact_navigation_block(optimization_inputs: Dict[str, Any]) -> str:
    """Workspace-relative paths the optimizer can use to read detailed data."""
    lines: List[str] = [
        "All paths are **relative to your workspace root**:",
        "",
        "- ``current_batch_trajectories.json`` — batch report with per-query metrics, ``evaluation_reason``, and ``trajectory_artifacts_relpath``.",
        "- ``<trajectory_artifacts_relpath>/working_trajectory/*.jsonl`` — full agent conversations (messages, tool calls, ``_tool_metadata``). Copy the relpath from each query row.",
        "- ``<trajectory_artifacts_relpath>/tool_call/tool_calls.jsonl`` — chronological audit log of all MCP tool calls.",
        "",
        "- ``mas_config/`` — MAS configuration (``config.yaml``, prompts, skills).",
    ]
    if _train_stats(optimization_inputs.get("train_performance", {})).get("backend") == "workbench":
        lines.extend([
            "", "**WorkBench action diagnostics:**",
            "- Compare ``predicted_actions`` with ``ground_truth_actions`` in each query row.",
            "- ``evaluation_reason`` includes state correctness and unwanted side effects.",
        ])
    return "\n".join(lines)


def build_task_specific_evaluation_guide(optimization_inputs: Dict[str, Any]) -> str:
    """Build task-specific evaluation guidance injected into backward_pass.md."""
    if _train_stats(optimization_inputs.get("train_performance", {})).get("backend") == "workbench":
        return _WORKBENCH_EVALUATION_GUIDE
    return "Use this benchmark's evaluation score and diagnostic fields to interpret training trajectories."


_WORKBENCH_EVALUATION_GUIDE = """\
   **WorkBench evaluation (state-based, official harness):**
   - Success requires predicted tool calls to produce the **same sandbox DB state** as the gold action list.
   - Check `evaluation_reason` for `unwanted_side_effects` (extra mutations) and empty `predicted_actions`.
   - Compare failed trajectories: did the agent search for IDs before delete/update? Wrong dates/times (9am-6pm rule)?
   - Optimize prompts/skills for search-then-act; do not add web/file tools (invalid for this benchmark).
"""


def trajectory_json_attachment_note(optimization_inputs: Dict[str, Any]) -> str:
    """Short reminder appended after the template body."""
    return (
        "\n\n**Data file:** ``current_batch_trajectories.json`` contains the batch report. "
        "Use ``queries[]`` for per-query details and ``trajectory_artifacts_relpath`` to locate JSONL logs."
    )
