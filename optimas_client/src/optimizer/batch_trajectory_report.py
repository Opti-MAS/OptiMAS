"""
Single-file batch report for ``current_batch_trajectories.json``.

Structure (flat — no unnecessary nesting):

* ``queries[]`` — one dict per query with *all* fields flat: identity, input, gold,
  run outcome (success, wall_time, output), evaluation (score, grader), and
  ``trajectory_artifacts_relpath`` pointing to JSONL logs.
* ``batch_aggregates`` — accuracy counters + wall-time stats + ``per_agent`` trajectory
  metrics + ``unique_tool_types_union``.  All at the same level (no nested ``efficiency``).

Per-agent metrics are collected from ``working_trajectory/*.jsonl`` on disk.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from .step_workspace import find_sample_workspace_dir, sample_workspace_basename

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# §1  Per-sample workspace resolution
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# §2  Trajectory JSONL parsing → per-query → batch aggregation
# ═══════════════════════════════════════════════════════════════════════════

_FNAME_NEW_RE = re.compile(r"^(.+)_([0-9a-f]{4,32})_(\d+)\.jsonl$")
_FNAME_OLD_RE = re.compile(r"^(.+)_(\d+)\.jsonl$")


def _parse_trajectory_filename(filename: str) -> Optional[Tuple[str, Optional[str], int]]:
    """Parse ``agent_name[_invocation_id]_phase.jsonl`` -> (agent, inv_id|None, phase)."""
    m = _FNAME_NEW_RE.match(filename)
    if m:
        return (m.group(1), m.group(2), int(m.group(3)))
    m = _FNAME_OLD_RE.match(filename)
    if m:
        return (m.group(1), None, int(m.group(2)))
    return None


def _parse_single_jsonl_stats(filepath: str) -> Dict[str, Any]:
    """Read one JSONL phase file and return per-phase statistics."""
    stats: Dict[str, Any] = {
        "assistant_turns": 0,
        "tool_calls": {},
        "skill_calls": {},
        "delegation_calls": {},
        "failed_tool_calls": 0,
    }
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or entry.get("type") == "tool_schema":
                    continue

                role = entry.get("role")

                if role == "assistant":
                    stats["assistant_turns"] += 1

                    for tc in entry.get("tool_calls") or []:
                        func = tc.get("function") or {}
                        name = func.get("name", "")
                        if not name:
                            continue
                        stats["tool_calls"][name] = stats["tool_calls"].get(name, 0) + 1

                        args = func.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                args = {}

                        if name == "load_skill":
                            sn = args.get("skill_name", "unknown") if isinstance(args, dict) else "unknown"
                            stats["skill_calls"][sn] = stats["skill_calls"].get(sn, 0) + 1
                        elif name == "assign_task_to_agent":
                            an = args.get("agent_name", "unknown") if isinstance(args, dict) else "unknown"
                            stats["delegation_calls"][an] = stats["delegation_calls"].get(an, 0) + 1

                elif role == "tool":
                    meta = entry.get("_tool_metadata") or {}
                    if meta.get("success") is False:
                        stats["failed_tool_calls"] += 1

    except OSError as e:
        logger.debug("Failed to read JSONL %s: %s", filepath, e)
    return stats


def _collect_query_trajectory_stats(working_traj_dir: str) -> Dict[str, Dict[str, Any]]:
    """Per-agent statistics from one query's ``working_trajectory/``."""
    if not os.path.isdir(working_traj_dir):
        return {}

    file_groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    try:
        for fname in os.listdir(working_traj_dir):
            if not fname.endswith(".jsonl"):
                continue
            parsed = _parse_trajectory_filename(fname)
            if parsed is None:
                continue
            agent_name, inv_id, phase = parsed
            key = (agent_name, inv_id or "_legacy")
            file_groups.setdefault(key, []).append(
                (phase, os.path.join(working_traj_dir, fname))
            )
    except OSError:
        return {}

    agent_stats: Dict[str, Dict[str, Any]] = {}

    for (agent_name, _inv_id), files in file_groups.items():
        if agent_name not in agent_stats:
            agent_stats[agent_name] = {
                "invocation_count": 0,
                "total_assistant_rounds": 0,
                "total_tool_calls": 0,
                "total_phases": 0,
                "tool_call_counts": {},
                "skill_call_counts": {},
                "delegation_counts": {},
                "total_failed_tool_calls": 0,
                "per_invocation_rounds": [],
                "per_invocation_tool_calls": [],
                "per_invocation_phases": [],
            }

        agg = agent_stats[agent_name]
        agg["invocation_count"] += 1

        inv_rounds = 0
        inv_tool_calls = 0
        inv_phases = len(files)

        for _phase, filepath in sorted(files, key=lambda x: x[0]):
            fs = _parse_single_jsonl_stats(filepath)
            tc_total = sum(fs["tool_calls"].values())

            inv_rounds += fs["assistant_turns"]
            inv_tool_calls += tc_total

            agg["total_assistant_rounds"] += fs["assistant_turns"]
            agg["total_tool_calls"] += tc_total
            agg["total_failed_tool_calls"] += fs["failed_tool_calls"]

            for tn, cnt in fs["tool_calls"].items():
                agg["tool_call_counts"][tn] = agg["tool_call_counts"].get(tn, 0) + cnt
            for sn, cnt in fs["skill_calls"].items():
                agg["skill_call_counts"][sn] = agg["skill_call_counts"].get(sn, 0) + cnt
            for dn, cnt in fs["delegation_calls"].items():
                agg["delegation_counts"][dn] = agg["delegation_counts"].get(dn, 0) + cnt

        agg["total_phases"] += inv_phases
        agg["per_invocation_rounds"].append(inv_rounds)
        agg["per_invocation_tool_calls"].append(inv_tool_calls)
        agg["per_invocation_phases"].append(inv_phases)

    return agent_stats


def _safe_avg(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0


def _compact_query_agent_stats(
    raw: Dict[str, Dict[str, Any]],
    configured_agents: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compress ``_collect_query_trajectory_stats`` output into a compact per-query summary.

    For each agent in this query, produces::

        {agent_name: {invocations, rounds, phases, tool_calls,
                      failed_tool_calls, tool_distribution, ...}}

    If *configured_agents* is given, agents that were never invoked for this
    query are included with ``invocations: 0`` so the optimizer can see them.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for agent_name, a in raw.items():
        row: Dict[str, Any] = {
            "invocations": a["invocation_count"],
            "rounds": a["total_assistant_rounds"],
            "phases": a["total_phases"],
            "tool_calls": a["total_tool_calls"],
            "failed_tool_calls": a["total_failed_tool_calls"],
            "tool_distribution": dict(sorted(a["tool_call_counts"].items())),
        }
        if a["skill_call_counts"]:
            row["skill_distribution"] = dict(sorted(a["skill_call_counts"].items()))
        if a["delegation_counts"]:
            row["delegation_distribution"] = dict(sorted(a["delegation_counts"].items()))
        out[agent_name] = row

    if configured_agents:
        for agent_name in configured_agents:
            if agent_name not in out:
                out[agent_name] = {"invocations": 0}

    return out


def _aggregate_batch_trajectory_stats(
    per_query_traj: Dict[str, Dict[str, Dict[str, Any]]],
    entry_agent_name: str,
    configured_agents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate pre-collected per-query trajectory stats into batch-level metrics.

    *per_query_traj* maps ``query_id → _collect_query_trajectory_stats()`` output.
    If *configured_agents* is given, agents that were never invoked across the
    entire batch are still included with zeroed-out metrics so the optimizer is
    explicitly aware of their inactivity.
    Returns ``{per_agent: {...}, unique_tool_types_union: [...]}``.
    """
    all_query_stats = list(per_query_traj.values())
    all_agents: set = set()
    for qs in all_query_stats:
        all_agents.update(qs.keys())
    if configured_agents:
        all_agents.update(configured_agents)

    if not all_query_stats and not configured_agents:
        return {}

    per_agent: Dict[str, Dict[str, Any]] = {}

    for agent_name in sorted(all_agents):
        is_entry = agent_name == entry_agent_name

        invocation_counts: List[int] = []
        all_inv_rounds: List[int] = []
        all_inv_tool_calls: List[int] = []
        all_inv_phases: List[int] = []
        batch_tool_counts: Dict[str, int] = {}
        batch_skill_counts: Dict[str, int] = {}
        batch_delegation_counts: Dict[str, int] = {}
        batch_failed: List[int] = []

        for qstats in all_query_stats:
            a = qstats.get(agent_name)
            if a is None:
                continue
            invocation_counts.append(a["invocation_count"])
            all_inv_rounds.extend(a["per_invocation_rounds"])
            all_inv_tool_calls.extend(a["per_invocation_tool_calls"])
            all_inv_phases.extend(a["per_invocation_phases"])
            batch_failed.append(a["total_failed_tool_calls"])

            for tn, cnt in a["tool_call_counts"].items():
                batch_tool_counts[tn] = batch_tool_counts.get(tn, 0) + cnt
            for sn, cnt in a["skill_call_counts"].items():
                batch_skill_counts[sn] = batch_skill_counts.get(sn, 0) + cnt
            for dn, cnt in a["delegation_counts"].items():
                batch_delegation_counts[dn] = batch_delegation_counts.get(dn, 0) + cnt

        n_queries = len(invocation_counts)
        n_invocations = sum(invocation_counts)

        if n_invocations == 0:
            per_agent[agent_name] = {
                "is_entry_agent": is_entry,
                "queries_appeared_in": 0,
                "total_invocations": 0,
                "note": "configured but never invoked in this batch",
            }
            continue

        agent_row: Dict[str, Any] = {
            "is_entry_agent": is_entry,
            "queries_appeared_in": n_queries,
            "avg_invocations_per_query": round(n_invocations / n_queries, 2),
            "max_invocations_per_query": max(invocation_counts),
            "avg_assistant_rounds": _safe_avg([float(x) for x in all_inv_rounds]),
            "max_assistant_rounds": max(all_inv_rounds) if all_inv_rounds else 0,
            "avg_phases": _safe_avg([float(x) for x in all_inv_phases]),
            "max_phases": max(all_inv_phases) if all_inv_phases else 0,
            "avg_tool_invocations": _safe_avg([float(x) for x in all_inv_tool_calls]),
            "max_tool_invocations": max(all_inv_tool_calls) if all_inv_tool_calls else 0,
            "avg_failed_tool_calls": round(sum(batch_failed) / n_queries, 2),
            "max_failed_tool_calls": max(batch_failed) if batch_failed else 0,
            "tool_names_observed": sorted(batch_tool_counts.keys()),
            "tool_call_distribution": {
                k: round(v / n_invocations, 2) for k, v in sorted(batch_tool_counts.items())
            },
        }
        if batch_skill_counts:
            agent_row["skill_call_distribution"] = {
                k: round(v / n_invocations, 2) for k, v in sorted(batch_skill_counts.items())
            }
        if batch_delegation_counts:
            agent_row["delegation_distribution"] = {
                k: round(v / n_invocations, 2) for k, v in sorted(batch_delegation_counts.items())
            }

        per_agent[agent_name] = agent_row

    all_tools: set = set()
    for m in per_agent.values():
        all_tools.update(m.get("tool_names_observed") or [])

    return {
        "per_agent": per_agent,
        "unique_tool_types_union": sorted(all_tools),
    }




# ═══════════════════════════════════════════════════════════════════════════
# §3  Report building
# ═══════════════════════════════════════════════════════════════════════════

_MAX_FINAL_MAS_OUTPUT_CHARS = 2_000_000


def _trunc(s: Any, max_len: int) -> str:
    if s is None:
        return ""
    text = s if isinstance(s, str) else str(s)
    if max_len <= 0 or len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _maybe_cap_full_text(s: str, cap: int) -> str:
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n\n[OUTPUT CAPPED AT {cap} CHARS FOR JSON SAFETY — FULL TEXT IS IN MAS TRACES]"


def compact_eval_result_row(
    eval_row: Dict[str, Any],
    *,
    max_prediction_len: int = 1200,
    max_grader_len: int = 1200,
    max_reasoning_len: int = 1200,
    mas_conversation_message_count: Optional[int] = None,
    optimizer_facing: bool = False,
) -> Dict[str, Any]:
    """Single row for ``current_batch_trajectories.json`` ``results`` list.

    *optimizer_facing* — when true, return only ``{score, grader_response, reasoning}``.
    """
    is_workbench = eval_row.get("backend") == "workbench"

    if optimizer_facing:
        base: Dict[str, Any] = {
            "score": eval_row.get("score"),
        }
        if is_workbench:
            base["correct"] = eval_row.get("correct", False)
            base["exact_match"] = eval_row.get("exact_match", False)
            base["unwanted_side_effects"] = eval_row.get("unwanted_side_effects", False)
            base["evaluation_reason"] = eval_row.get("evaluation_reason", "")
            pred = eval_row.get("predicted_actions") or []
            gold = eval_row.get("ground_truth_actions") or []
            base["predicted_actions_count"] = len(pred)
            base["ground_truth_actions_count"] = len(gold)
        else:
            base["grader_response"] = _trunc(eval_row.get("grader_response", ""), max_grader_len)
            base["reasoning"] = _trunc(eval_row.get("reasoning", ""), max_reasoning_len)
        return base

    out: Dict[str, Any] = {
        "problem_id": eval_row.get("problem_id"),
        "score": eval_row.get("score"),
        "backend": eval_row.get("backend"),
    }
    if is_workbench:
        out["correct"] = eval_row.get("correct", False)
        out["exact_match"] = eval_row.get("exact_match", False)
        out["unwanted_side_effects"] = eval_row.get("unwanted_side_effects", False)
        out["evaluation_reason"] = eval_row.get("evaluation_reason", "")
        out["predicted_actions"] = eval_row.get("predicted_actions") or []
        out["ground_truth_actions"] = eval_row.get("ground_truth_actions") or []
        out["expected_answer"] = eval_row.get("expected_answer")
    else:
        out["extracted_answer"] = eval_row.get("extracted_answer")
        out["expected_answer"] = eval_row.get("expected_answer")
        out["prediction"] = _trunc(eval_row.get("prediction", ""), max_prediction_len)
        out["grader_response"] = _trunc(eval_row.get("grader_response", ""), max_grader_len)
        out["reasoning"] = _trunc(eval_row.get("reasoning", ""), max_reasoning_len)
    if mas_conversation_message_count is not None:
        out["mas_conversation_message_count"] = int(mas_conversation_message_count)
    return out


def _query_eval_score(q: Dict[str, Any]) -> float:
    """Score from a flat query dict (new format) or nested ``evaluation`` (legacy)."""
    score = q.get("score")
    if score is None:
        score = (q.get("evaluation") or {}).get("score")
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sync_correct_answer_ids_in_aggregates(
    batch_aggregates: Dict[str, Any], queries: List[Dict[str, Any]]
) -> None:
    ids = [
        str(q["query_id"])
        for q in queries
        if q.get("query_id") is not None and _query_eval_score(q) > 0.0
    ]
    batch_aggregates["correct_answer_ids"] = ids
    batch_aggregates["correct_answer_count"] = len(ids)


def build_batch_report(
    batch: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]],
    forward_results: List[Dict[str, Any]],
    train_stats: Dict[str, Any],
    *,
    entry_agent_name: str,
    step_dir: Optional[str] = None,
    configured_agents: Optional[List[str]] = None,
    max_grader_len: int = 1200,
    max_reasoning_len: int = 1200,
    resampled_query_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the batch report with flat per-query dicts and flat batch_aggregates.

    *configured_agents*: all agent names from the MAS config (entry + sub).
    When provided, agents that were never invoked still appear in per-query
    ``agent_stats`` and batch-level ``per_agent`` with ``invocations: 0``.

    *resampled_query_ids*: query IDs that were re-sampled from the previous
    step's failures (adaptive sampling).  Each matching query gets an
    ``is_resampled: true`` flag in the report.
    """
    n = len(batch)
    if len(eval_results) != n or len(forward_results) != n:
        raise ValueError(
            f"Alignment error: batch={n}, eval={len(eval_results)}, forward={len(forward_results)}"
        )
    if step_dir is not None:
        step_dir = os.path.abspath(step_dir)

    _resampled_set: set = set(resampled_query_ids or [])

    queries: List[Dict[str, Any]] = []
    per_query_traj: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for i in range(n):
        problem = batch[i] or {}
        ev = eval_results[i] or {}
        fr = forward_results[i] or {}
        qid = str(
            ev.get("problem_id") or problem.get("query_id") or problem.get("id") or f"idx-{i}"
        )

        wt = fr.get("wall_time_sec")
        wt_val = round(float(wt), 2) if isinstance(wt, (int, float)) else None

        data = fr.get("data")
        final_output = _maybe_cap_full_text(
            data if isinstance(data, str) else str(data), _MAX_FINAL_MAS_OUTPUT_CHARS
        ) if data is not None else None

        relp = sample_workspace_basename(step_dir, qid) if step_dir else None

        # per-query agent stats from JSONL logs
        agent_stats: Optional[Dict[str, Any]] = None
        if step_dir and relp:
            traj_dir = os.path.join(step_dir, relp, "working_trajectory")
            raw = _collect_query_trajectory_stats(traj_dir)
            if raw:
                per_query_traj[qid] = raw
                agent_stats = _compact_query_agent_stats(raw, configured_agents)
            elif configured_agents:
                agent_stats = _compact_query_agent_stats({}, configured_agents)

        is_workbench_eval = ev.get("backend") == "workbench"

        qobj: Dict[str, Any] = {
            "query_id": qid,
            "query": (problem.get("query") or problem.get("question") or "").strip(),
            "query_type": str(problem.get("query_type") or problem.get("topic") or "").strip(),
            "trajectory_artifacts_relpath": relp,
            "success": bool(fr.get("success", False)),
            "error": fr.get("error") or None,
            "wall_time_sec": wt_val,
            "final_mas_output": final_output,
            "score": ev.get("score"),
            "is_resampled": qid in _resampled_set,
        }

        if is_workbench_eval:
            qobj["evaluation_reason"] = ev.get("evaluation_reason", "")
            qobj["correct"] = ev.get("correct", False)
            qobj["exact_match"] = ev.get("exact_match", False)
            qobj["unwanted_side_effects"] = ev.get("unwanted_side_effects", False)
            predicted = ev.get("predicted_actions") or []
            gold = ev.get("ground_truth_actions") or []
            qobj["predicted_actions_count"] = len(predicted)
            qobj["ground_truth_actions_count"] = len(gold)
            qobj["predicted_actions"] = predicted
            qobj["ground_truth_actions"] = gold
            qobj["gold_answer"] = ev.get("expected_answer")
        else:
            qobj["gold_answer"] = ev.get("expected_answer")
            qobj["grader_response"] = _trunc(ev.get("grader_response", ""), max_grader_len)
            qobj["reasoning"] = _trunc(ev.get("reasoning", ""), max_reasoning_len)

        if agent_stats:
            qobj["agent_stats"] = agent_stats
        queries.append(qobj)

    # -- batch aggregates (flat) --
    walls = [q["wall_time_sec"] for q in queries if q["wall_time_sec"] is not None]
    batch_aggregates: Dict[str, Any] = {
        **dict(train_stats),
        "avg_mas_wall_time_sec": round(sum(walls) / len(walls), 2) if walls else None,
        "max_mas_wall_time_sec": round(max(walls), 2) if walls else None,
    }

    if step_dir:
        try:
            traj = _aggregate_batch_trajectory_stats(
                per_query_traj, entry_agent_name, configured_agents,
            )
            batch_aggregates.update(traj)
        except Exception as e:
            logger.warning("batch trajectory aggregation failed: %s", e)

    _sync_correct_answer_ids_in_aggregates(batch_aggregates, queries)

    resampled_count = sum(1 for q in queries if q.get("is_resampled"))
    if resampled_count > 0:
        batch_aggregates["resampled_count"] = resampled_count
        batch_aggregates["resampled_query_ids"] = [
            q["query_id"] for q in queries if q.get("is_resampled")
        ]

    return {
        "batch_aggregates": batch_aggregates,
        "entry_agent": entry_agent_name,
        "queries": queries,
    }


def build_batch_report_from_train_performance(
    train_perf: Dict[str, Any],
    *,
    entry_agent_name: str = "worker",
) -> Optional[Dict[str, Any]]:
    """If ``batch_report`` is already present, return it; else rebuild from legacy keys."""
    br = train_perf.get("batch_report")
    if isinstance(br, dict) and br.get("queries"):
        br2 = copy.deepcopy(br)
        ba = br2.get("batch_aggregates")
        if isinstance(ba, dict):
            _sync_correct_answer_ids_in_aggregates(ba, br2.get("queries") or [])
        return br2

    batch = train_perf.get("batch") or []
    evs = train_perf.get("results") or []
    stats = train_perf.get("stats") or {}

    n = len(batch)
    if n == 0:
        return None
    forward_results = [{"success": False, "data": None, "error": None, "wall_time_sec": None} for _ in range(n)]
    try:
        return build_batch_report(
            batch, evs, forward_results, stats,
            entry_agent_name=entry_agent_name, step_dir=None,
        )
    except Exception as e:
        logger.warning("build_batch_report_from_train_performance: %s", e)
        return None


__all__ = [
    "build_batch_report",
    "build_batch_report_from_train_performance",
    "compact_eval_result_row",
    "find_sample_workspace_dir",
    "sample_workspace_basename",
]
