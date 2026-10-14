"""
Utility functions for the MAS training loop.

Stateless helpers for:
- Resume-from-checkpoint: parsing, state restoration, cleanup
- Optimizer long-term memory: archiving working logs, carrying forward
- Metrics persistence: run_metrics.json, run_summary.md, optimization_history.json
- Evaluation aggregation: accuracy computation, failure categorisation

All functions use only the Python standard library so they can be tested and
invoked independently of the heavy training imports (MCP, LLM, etc.).
"""

import json
import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_accuracy(evals: List[Dict[str, Any]]) -> float:
    if not evals:
        return 0.0
    return sum(e.get("score", 0.0) for e in evals) / len(evals)


def build_aggregated_stats(
    evals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build aggregated statistics from evaluation results.

    Returns a dict with failure categorisation and accuracy — designed to give
    the optimizer a structured diagnostic view.  Per-agent turn / tool
    statistics are computed separately in ``build_batch_report`` from JSONL
    trajectory files (the in-memory ``conversation_history`` is incomplete due
    to ``phase_done`` resets and sub-agent isolation).
    """
    total = len(evals)
    if total == 0:
        return {"total": 0}

    successes = [e for e in evals if e.get("score", 0) > 0]
    failures = [e for e in evals if e.get("score", 0) <= 0]
    correct_answer_ids = [e.get("problem_id") for e in successes]

    is_workbench = any(e.get("backend") == "workbench" for e in evals)

    if is_workbench:
        return _build_workbench_stats(evals, total, successes, failures, correct_answer_ids)

    agent_errors = []
    wrong_answers = []
    grader_errors = []
    for e in failures:
        if e.get("grader_error"):
            grader_errors.append(e.get("problem_id"))
        elif e.get("prediction", "").startswith("[Agent error]") or e.get("prediction", "").startswith("Error:"):
            agent_errors.append(e.get("problem_id"))
        else:
            wrong_answers.append(e.get("problem_id"))

    return {
        "total": total,
        "success_count": len(successes),
        "failure_count": len(failures),
        "accuracy": round(len(successes) / total, 4) if total else 0,
        "agent_error_ids": agent_errors,
        "wrong_answer_ids": wrong_answers,
        "correct_answer_ids": correct_answer_ids,
        "grader_error_ids": grader_errors,
        "agent_error_count": len(agent_errors),
        "wrong_answer_count": len(wrong_answers),
        "correct_answer_count": len(correct_answer_ids),
        "grader_error_count": len(grader_errors),
    }


def _build_workbench_stats(
    evals: List[Dict[str, Any]],
    total: int,
    successes: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    correct_answer_ids: List[str],
) -> Dict[str, Any]:
    """Build WorkBench aggregate statistics (official state-based correctness)."""
    side_effect_ids = [e.get("problem_id") for e in failures if e.get("unwanted_side_effects")]
    no_action_ids = []
    agent_errors = []
    wrong_state_ids = []
    for e in failures:
        pred = e.get("predicted_actions") or []
        if not pred and not e.get("success", True):
            if str(e.get("prediction", "")).startswith("[Agent error]"):
                agent_errors.append(e.get("problem_id"))
            else:
                no_action_ids.append(e.get("problem_id"))
        elif e.get("problem_id") not in side_effect_ids:
            wrong_state_ids.append(e.get("problem_id"))

    return {
        "total": total,
        "backend": "workbench",
        "success_count": len(successes),
        "failure_count": len(failures),
        "accuracy": round(len(successes) / total, 4) if total else 0,
        "correct_answer_ids": correct_answer_ids,
        "correct_answer_count": len(correct_answer_ids),
        "unwanted_side_effects_count": len(side_effect_ids),
        "unwanted_side_effects_ids": side_effect_ids,
        "no_action_count": len(no_action_ids),
        "no_action_ids": no_action_ids,
        "agent_error_count": len(agent_errors),
        "agent_error_ids": agent_errors,
        "wrong_state_count": len(wrong_state_ids),
        "wrong_state_ids": wrong_state_ids,
    }


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def save_metrics(run_metrics: Dict[str, Any], workspace_dir: str) -> None:
    metrics_path = os.path.join(workspace_dir, "run_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(run_metrics, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(workspace_dir, "run_summary.md")
    sp = run_metrics.get("split") or {}
    lines = [
        "# Run Summary",
        "",
        f"- **Task ID**: {run_metrics['task_id']}",
        f"- **Dataset**: {run_metrics['dataset_path']}",
        f"- **Epochs**: {run_metrics['max_epochs']}, **Batch size**: {run_metrics['batch_size']}",
        "",
        f"- **Total**: {sp.get('total', '?')} | **Train**: {sp.get('train', '?')} "
        f"| **Val**: {sp.get('val', '?')} | **Test**: {sp.get('test', '?')}",
        "",
        "## Steps",
        "",
        "| Epoch | Step | Train Acc | Val Acc | Val Diff |",
        "|-------|------|-----------|---------|----------|",
    ]
    for s in run_metrics["steps"]:
        if "error" in s:
            lines.append(f"| {s['epoch']} | {s['step']} | ERROR | ERROR | - |")
        else:
            lines.append(
                f"| {s['epoch']} | {s['step']} | {s['train_acc']:.4f} "
                f"| {s['val_acc']:.4f} | {s['val_acc_diff']:+.4f} |"
            )
    bv = run_metrics.get("best_val")
    lines.extend(["", "## Best Validation Config", ""])
    if bv and bv.get("val_acc") is not None:
        lines.append(f"- **Best Val Accuracy**: {bv['val_acc']:.4f}")
        lines.append(f"- **Config**: `{bv.get('config_dir', 'N/A')}`")
    else:
        lines.append("- No valid config recorded.")
    lines.append("")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Optimization history
# ---------------------------------------------------------------------------

HISTORY_FILENAME = "optimization_history.json"


def _build_agent_stats(per_agent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a list of per-agent stat dicts from batch_aggregates.per_agent."""
    if not per_agent:
        return []
    results: List[Dict[str, Any]] = []
    sorted_agents = sorted(
        per_agent.keys(),
        key=lambda a: (0 if (per_agent[a] or {}).get("is_entry_agent") else 1, a),
    )
    for agent_name in sorted_agents:
        m = per_agent[agent_name]
        if not isinstance(m, dict):
            continue

        if m.get("total_invocations") == 0 or m.get("queries_appeared_in") == 0:
            results.append({
                "agent": agent_name,
                "role": "entry" if m.get("is_entry_agent") else "sub-agent",
                "total_invocations": 0,
                "note": m.get("note", "configured but never invoked"),
            })
            continue

        entry: Dict[str, Any] = {
            "agent": agent_name,
            "role": "entry" if m.get("is_entry_agent") else "sub-agent",
            "avg_rounds": m.get("avg_assistant_rounds"),
            "avg_phases": m.get("avg_phases"),
            "avg_tool_calls": m.get("avg_tool_invocations"),
        }
        if not m.get("is_entry_agent"):
            entry["avg_invocations_per_query"] = m.get("avg_invocations_per_query")
        if m.get("tool_call_distribution"):
            entry["tool_distribution"] = m["tool_call_distribution"]
        if m.get("skill_call_distribution"):
            entry["skill_distribution"] = m["skill_call_distribution"]
        if m.get("delegation_distribution"):
            entry["delegation_distribution"] = m["delegation_distribution"]
        results.append(entry)
    return results


def append_optimization_history(
    opt_dir: str,
    step_number: int,
    epoch: int,
    train_acc: float,
    val_acc: float,
    val_diff: float,
    batch_size: int,
    batch_aggregates: Dict[str, Any],
    val_stats: Dict[str, Any],
    optimization_summary: str,
) -> None:
    """Append one step record to optimization_history.json."""
    history_path = os.path.join(opt_dir, HISTORY_FILENAME)

    # Load existing history or start fresh.  The optimizer agent may have
    # already written this file as a flat JSON array instead of the canonical
    # ``{"steps": [...]}`` dict — handle both formats gracefully.
    steps: List[Dict[str, Any]] = []
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                steps = data.get("steps", [])
            elif isinstance(data, list):
                steps = data
            else:
                logger.warning("Unexpected type in %s — starting fresh", history_path)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt %s — starting fresh", history_path)

    per_agent = batch_aggregates.get("per_agent", {})
    avg_wall = batch_aggregates.get("avg_mas_wall_time_sec")

    summary_short = optimization_summary.replace("\n", " ").strip()
    if len(summary_short) > 500:
        summary_short = summary_short[:497] + "..."

    step_record: Dict[str, Any] = {
        "step": step_number,
        "epoch": epoch,
        "train_accuracy": round(train_acc, 4),
        "train_correct": batch_aggregates.get("correct_answer_count"),
        "train_total": batch_aggregates.get("total", batch_size),
        "val_accuracy": round(val_acc, 4),
        "val_correct": val_stats.get("success_count"),
        "val_total": val_stats.get("total"),
        "val_delta": round(val_diff, 4),
        "avg_wall_time_sec": round(avg_wall, 1) if avg_wall is not None else None,
    }

    step_record["errors"] = {
        "wrong": batch_aggregates.get("wrong_answer_count"),
        "agent_error": batch_aggregates.get("agent_error_count", 0),
    }

    step_record["agent_stats"] = _build_agent_stats(per_agent)
    step_record["summary"] = summary_short

    steps.append(step_record)

    history = {"steps": steps}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Optimizer long-term memory
# ---------------------------------------------------------------------------

def archive_optimizer_working_log(opt_dir: str, step_number: int) -> None:
    """Archive optimizer working files (excluding working_trajectory/ and step_*_working_log/)
    into a new step_<N>_working_log/ folder within opt_dir."""
    archive_name = f"step_{step_number}_working_log"
    archive_dir = os.path.join(opt_dir, archive_name)
    if os.path.exists(archive_dir):
        return

    _SKIP = {"working_trajectory", "optimizer_context_bundle.json"}
    _PERSISTENT_FILES = {"optimization_history.json", "hypothesis_ledger.md"}
    items_to_archive = []
    try:
        for name in os.listdir(opt_dir):
            if name in _SKIP:
                continue
            if name.startswith("step_") and name.endswith("_working_log"):
                continue
            if name in _PERSISTENT_FILES:
                continue
            items_to_archive.append(name)
    except OSError:
        return

    if not items_to_archive:
        return

    os.makedirs(archive_dir, exist_ok=True)
    for name in items_to_archive:
        src = os.path.join(opt_dir, name)
        dst = os.path.join(archive_dir, name)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        except OSError as e:
            logger.warning("Failed to archive %s: %s", src, e)


def carry_forward_optimizer_memory(prev_opt_dir: str, new_opt_dir: str) -> None:
    """Copy step_*_working_log/ folders and optimization_history.json from previous
    optimizer dir to the new one, building long-term memory across steps."""
    if not os.path.isdir(prev_opt_dir):
        return
    _PERSISTENT_FILES = {"optimization_history.json", "hypothesis_ledger.md"}
    try:
        for name in os.listdir(prev_opt_dir):
            if name.startswith("step_") and name.endswith("_working_log"):
                src = os.path.join(prev_opt_dir, name)
                dst = os.path.join(new_opt_dir, name)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
            elif name in _PERSISTENT_FILES:
                src = os.path.join(prev_opt_dir, name)
                dst = os.path.join(new_opt_dir, name)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
    except OSError as e:
        logger.warning("Failed to carry forward optimizer memory: %s", e)


# ---------------------------------------------------------------------------
# Resume-from-checkpoint helpers
# ---------------------------------------------------------------------------

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")
_EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")


def parse_resume_checkpoint(resume_from: str) -> Dict[str, Any]:
    """Parse a resume checkpoint path and extract training state.

    *resume_from* must point to a valid step directory, either as an absolute
    path or relative to cwd.  The directory must contain ``mas_config/`` and
    ``optimizer/``.

    Returns a dict with keys:
        workspace_dir, epoch, step, step_dir, mas_config_dir, opt_dir
    """
    step_dir = os.path.abspath(resume_from)
    if not os.path.isdir(step_dir):
        raise ValueError(f"Resume checkpoint directory does not exist: {resume_from}")

    step_name = os.path.basename(step_dir)
    m = _STEP_DIR_RE.match(step_name)
    if not m:
        raise ValueError(
            f"Cannot parse step number from directory name '{step_name}'. "
            "Expected format: step_<N>"
        )
    resume_step = int(m.group(1))

    epoch_dir = os.path.dirname(step_dir)
    epoch_name = os.path.basename(epoch_dir)
    em = _EPOCH_DIR_RE.match(epoch_name)
    if not em:
        raise ValueError(
            f"Cannot parse epoch number from directory name '{epoch_name}'. "
            "Expected format: epoch_<N>"
        )
    resume_epoch = int(em.group(1))

    workspace_dir = os.path.dirname(epoch_dir)

    mas_config_dir = os.path.join(step_dir, "mas_config")
    if not os.path.isdir(mas_config_dir):
        raise ValueError(f"mas_config/ not found in resume checkpoint: {step_dir}")

    opt_dir = os.path.join(step_dir, "optimizer")
    if not os.path.isdir(opt_dir):
        raise ValueError(f"optimizer/ not found in resume checkpoint: {step_dir}")

    return {
        "workspace_dir": workspace_dir,
        "epoch": resume_epoch,
        "step": resume_step,
        "step_dir": step_dir,
        "mas_config_dir": mas_config_dir,
        "opt_dir": opt_dir,
    }


def restore_prev_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Read saved artefacts from the checkpoint and reconstruct loop variables.

    Returns a dict with: prev_optimization_summary, prev_val_acc,
    best_val_acc, best_val_config_dir, run_metrics.
    """
    opt_dir = ckpt["opt_dir"]
    workspace_dir = ckpt["workspace_dir"]
    resume_step = ckpt["step"]

    details_path = os.path.join(opt_dir, "optimization_details.md")
    if os.path.isfile(details_path):
        with open(details_path, "r", encoding="utf-8") as f:
            prev_optimization_summary = f.read()
    else:
        prev_optimization_summary = "Resumed from checkpoint; no previous summary available."

    metrics_path = os.path.join(workspace_dir, "run_metrics.json")
    prev_val_acc = 0.0
    best_val_acc = -1.0
    best_val_config_dir: Optional[str] = None
    run_metrics: Optional[Dict[str, Any]] = None

    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            run_metrics = json.load(f)
        valid_steps = [s for s in run_metrics.get("steps", []) if s.get("step", 0) <= resume_step]
        run_metrics["steps"] = valid_steps
        for s in valid_steps:
            vacc = s.get("val_acc")
            if vacc is None or "error" in s:
                continue
            if s["step"] == resume_step:
                prev_val_acc = vacc
            if vacc >= best_val_acc:
                best_val_acc = vacc
                best_val_config_dir = s.get("evaluated_config_dir")
                if not best_val_config_dir:
                    raise ValueError("Checkpoint lacks an evaluated configuration snapshot; its validation score cannot be attributed safely.")

    return {
        "prev_optimization_summary": prev_optimization_summary,
        "prev_val_acc": prev_val_acc,
        "best_val_acc": best_val_acc,
        "best_val_config_dir": best_val_config_dir,
        "run_metrics": run_metrics,
    }


def cleanup_steps_after(workspace_dir: str, resume_epoch: int, resume_step: int) -> None:
    """Delete all step directories (and their validation counterparts) after
    the resume checkpoint, as well as any empty epoch directories beyond the
    resume epoch.
    """
    deleted = []
    for epoch_name in sorted(os.listdir(workspace_dir)):
        em = _EPOCH_DIR_RE.match(epoch_name)
        if not em:
            continue
        epoch_num = int(em.group(1))
        epoch_path = os.path.join(workspace_dir, epoch_name)
        if not os.path.isdir(epoch_path):
            continue

        for item in sorted(os.listdir(epoch_path)):
            sm = re.match(r"^step_(\d+)(_validation)?$", item)
            if not sm:
                continue
            step_num = int(sm.group(1))
            if epoch_num > resume_epoch or (epoch_num == resume_epoch and step_num > resume_step):
                target = os.path.join(epoch_path, item)
                try:
                    shutil.rmtree(target)
                    deleted.append(f"{epoch_name}/{item}")
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", target, e)

        if epoch_num > resume_epoch:
            remaining = os.listdir(epoch_path)
            if not remaining:
                try:
                    os.rmdir(epoch_path)
                    deleted.append(epoch_name)
                except OSError:
                    pass

    if deleted:
        logger.info("Cleaned up %d directories after checkpoint: %s", len(deleted), ", ".join(deleted[:10]))
