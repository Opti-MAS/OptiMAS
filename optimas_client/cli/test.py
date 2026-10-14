"""
MAS Configuration Tester with pass@N and mean@N

Evaluates a MAS config folder on a dataset with repeated trials per query,
producing robust accuracy metrics that account for agent stochasticity.

Metrics:
  - mean@N: average accuracy across N independent runs per query
  - pass@N: fraction of queries where at least one of N runs is correct
"""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..src.data.dataloader import UniversalDataLoader
from ..src.evaluation.evaluator import UniversalEvaluator
from ..src.engine.forward_engine import MASForwardEngine
from ..src.config.loader import ConfigLoader
from ..src.llm.factory import create_llm
from ..src.data.dataloader import (
    DEFAULT_TRAIN_VAL_TEST_RATIOS,
    DEFAULT_DETERMINISTIC_SPLIT_SEED,
)

logger = logging.getLogger(__name__)

_TRAIN_RUN_TIMESTAMP_SUFFIX = re.compile(r"-train-(\d{8}_\d{6})$")
_EPOCH_STEP_PATTERN = re.compile(r"(epoch_\d+)/(step_\d+)")


def _is_path_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

def _extract_train_timestamp_suffix_from_mas_config(
    mas_config_path: str,
    workspace_root: str,
) -> Optional[str]:
    """If ``mas_config`` lives under workspace inside a ``*-train-YYYYMMDD_HHMMSS`` run dir, return that suffix.

    This keeps ``{task_id}-test-{suffix}`` aligned with ``{task_id}-train-{suffix}`` when evaluating a
    checkpoint from a training workspace. Otherwise returns ``None`` (caller uses wall-clock time).
    """
    try:
        ws = Path(workspace_root).expanduser().resolve()
        mas = Path(mas_config_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    if not mas.exists():
        return None
    cur = mas.parent if mas.is_file() else mas
    for ancestor in [cur, *cur.parents]:
        if not _is_path_under_root(ancestor, ws):
            break
        m = _TRAIN_RUN_TIMESTAMP_SUFFIX.search(ancestor.name)
        if m:
            return m.group(1)
    return None

def _extract_config_source_suffix(
    mas_config_path: str,
    workspace_root: str,
) -> str:
    """Derive a disambiguating suffix from the MAS config's location within a training workspace.

    Returns e.g. ``-epoch_0_step_7`` for checkpoint configs, ``-best`` for best_mas_config,
    or ``""`` for configs not under a training workspace (e.g. initial configs in optimas_client).
    """
    try:
        mas = Path(mas_config_path).expanduser().resolve()
        ws = Path(workspace_root).expanduser().resolve()
    except (OSError, RuntimeError):
        return ""
    if not mas.exists():
        return ""

    cur = mas.parent if mas.is_file() else mas
    train_run_dir = None
    for ancestor in [cur, *cur.parents]:
        if not _is_path_under_root(ancestor, ws):
            break
        if _TRAIN_RUN_TIMESTAMP_SUFFIX.search(ancestor.name):
            train_run_dir = ancestor
            break

    if train_run_dir is None:
        return ""

    try:
        rel_str = str(mas.relative_to(train_run_dir))
    except ValueError:
        return ""

    if "best_mas_config" in rel_str:
        return "-best"

    m = _EPOCH_STEP_PATTERN.search(rel_str)
    if m:
        return f"-{m.group(1)}_{m.group(2)}"

    return ""

def _compute_mean_at_n(per_query_scores: Dict[str, List[float]]) -> float:
    """mean@N: for each query, average score across N runs, then average over queries."""
    if not per_query_scores:
        return 0.0
    query_means = [sum(scores) / len(scores) for scores in per_query_scores.values() if scores]
    return sum(query_means) / len(query_means) if query_means else 0.0


def _compute_pass_at_n(per_query_scores: Dict[str, List[float]]) -> float:
    """pass@N: fraction of queries where at least one run scored > 0."""
    if not per_query_scores:
        return 0.0
    passed = sum(1 for scores in per_query_scores.values() if any(s > 0 for s in scores))
    return passed / len(per_query_scores)


async def run_test(
    dataset_path: Optional[str] = None,
    mas_config_path: str = "",
    mcp_server_url: str = "",
    mcp_api_key: str = "",
    task_id: str = "",
    config_path: str = "optimas_client/config/global_config.yaml",
    optimizer_config_path: str = "optimas_client/config/optimizer_workbench/config.yaml",
    n_trials: int = 1,
    batch_size: int = 32,
    max_mcp_concurrent: Optional[int] = None,
    split: str = "test",
    split_ratios: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    train_dataset_path: Optional[str] = None,
    val_dataset_path: Optional[str] = None,
    test_dataset_path: Optional[str] = None,
    seed: int = DEFAULT_DETERMINISTIC_SPLIT_SEED,
):
    """
    Run MAS evaluation with pass@N and mean@N.

    Args:
        dataset_path: Path to JSONL dataset.
        mas_config_path: Path to the MAS config directory to evaluate.
        mcp_server_url: MCP server URL.
        mcp_api_key: MCP API key.
        task_id: Unique identifier for this test run.
        config_path: Path to global config yaml.
        optimizer_config_path: Path to optimizer config yaml.
        n_trials: Number of independent runs per query (N in pass@N / mean@N).
        batch_size: Number of queries to evaluate per forward batch.
        max_mcp_concurrent: Max concurrent MCP sessions.
        split: Which data split to use ("test", "val", "train", "all").
        output_dir: Directory to save results. Defaults to workspace_dir/task_id.
    """
    logger.info(f"Starting MAS Test (N={n_trials}, batch_size={batch_size}, split={split})")
    if n_trials < 1 or batch_size < 1:
        raise ValueError("n_trials and batch_size must be positive.")

    # ── 1. Load configs ──────────────────────────────────────────────────────
    global_config = ConfigLoader.load_config(config_path=config_path)
    optimizer_config_data = ConfigLoader.load_config(config_path=optimizer_config_path)
    global_config.update(optimizer_config_data)
    mas_config = ConfigLoader.load_config(mas_config_path)
    if not dataset_path and not (train_dataset_path or val_dataset_path or test_dataset_path):
        benchmark = global_config.get("optimizer", {}).get("benchmark", {})
        if split in ("train", "all"):
            train_dataset_path = benchmark.get("train_dataset_file_path")
        if split in ("val", "all"):
            val_dataset_path = benchmark.get("val_dataset_file_path")
        if split in ("test", "all"):
            test_dataset_path = benchmark.get("test_dataset_file_path")

    # ── 2. DataLoader ────────────────────────────────────────────────────────
    use_presplit = bool(train_dataset_path or val_dataset_path or test_dataset_path)
    if use_presplit:
        dataloader = UniversalDataLoader(
            seed=seed,
            train_path=train_dataset_path,
            val_path=val_dataset_path,
            test_path=test_dataset_path,
        )
    else:
        if dataset_path is None:
            raise ValueError(
                "Either --dataset or --train-dataset/--val-dataset/--test-dataset must be provided."
            )
        if split_ratios is None:
            split_ratios = list(DEFAULT_TRAIN_VAL_TEST_RATIOS)
        dataloader = UniversalDataLoader(
            dataset_path=dataset_path, seed=seed,
            train_ratio=split_ratios[0], val_ratio=split_ratios[1], test_ratio=split_ratios[2],
        )

    if split == "test":
        dataset = dataloader.get_test_set()
    elif split == "val":
        dataset = dataloader.get_validation_set()
    elif split == "train":
        dataset = dataloader.train_set
    elif split == "all":
        dataset = dataloader.all_data
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'test', 'val', 'train', or 'all'.")

    if not dataset:
        logger.error(f"Empty dataset for split={split!r}. Aborting.")
        raise ValueError(f"Empty dataset for split={split!r}.")

    logger.info(f"Loaded {len(dataset)} examples from split={split!r}")

    # ── 3. Evaluator ─────────────────────────────────────────────────────────
    optimizer_config = global_config.get("optimizer", {})
    grader_llm_config = optimizer_config.get("benchmark", {}).get("grader_llm")
    grader_llm = None
    if grader_llm_config:
        grader_llm = create_llm(
            grader_llm_config,
            ConfigLoader.get_api_recording_config(global_config),
            ConfigLoader.get_llm_settings(global_config),
        )
    evaluator = UniversalEvaluator(grader_llm=grader_llm)

    # ── 4. Forward engine ────────────────────────────────────────────────────
    workspace_root = global_config.get("workspace_base_dir", "./workspace")
    concurrent = max_mcp_concurrent if max_mcp_concurrent is not None else min(len(dataset), 8)
    per_problem_timeout = global_config.get("per_problem_timeout")
    engine = MASForwardEngine(
        mcp_server_url=mcp_server_url,
        mcp_api_key=mcp_api_key,
        workspace_root=workspace_root,
        max_concurrent=concurrent,
        per_problem_timeout=per_problem_timeout,
        test_mode=True,
    )

    # ── 5. Output directory ──────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(workspace_root, task_id)
    os.makedirs(output_dir, exist_ok=True)

    # ── 5.1 Save a copy of the MAS config used for this test ─────────────
    mas_config_src = Path(mas_config_path).expanduser().resolve()
    mas_config_snapshot_dir = os.path.join(output_dir, "mas_config")
    if mas_config_src.is_dir():
        shutil.copytree(str(mas_config_src), mas_config_snapshot_dir, dirs_exist_ok=True)
    elif mas_config_src.is_file():
        os.makedirs(mas_config_snapshot_dir, exist_ok=True)
        shutil.copy2(str(mas_config_src), mas_config_snapshot_dir)
    logger.info("MAS config snapshot saved to %s", mas_config_snapshot_dir)

    # ── 6. Run N trials ──────────────────────────────────────────────────────
    # per_query_scores[query_id] = [score_trial_0, score_trial_1, ...]
    per_query_scores: Dict[str, List[float]] = {
        p.get("query_id", f"q_{i}"): [] for i, p in enumerate(dataset)
    }
    all_trial_details: List[Dict[str, Any]] = []

    total_t0 = time.time()
    for trial in range(n_trials):
        trial_label = f"trial_{trial}"
        logger.info(f"=== Trial {trial}/{n_trials} ===")

        trial_t0 = time.time()
        trial_per_query: List[Dict[str, Any]] = []
        trial_acc_sum = 0.0
        processed = 0

        for batch_start in range(0, len(dataset), max(1, batch_size)):
            batch_end = min(batch_start + max(1, batch_size), len(dataset))
            batch = dataset[batch_start:batch_end]
            batch_label = f"{trial_label}/batch_{batch_start // max(1, batch_size)}"
            ws_subdir = f"{task_id}/{batch_label}"

            logger.info(
                "Forward batch %d-%d/%d (trial %d/%d)",
                batch_start + 1,
                batch_end,
                len(dataset),
                trial,
                n_trials,
            )
            batch_t0 = time.time()
            results = await engine.forward(batch, mas_config, workspace_subdir=ws_subdir, global_config=global_config)
            batch_fwd_time = time.time() - batch_t0
            logger.info("Batch forward done in %.1fs", batch_fwd_time)

            evals = await evaluator.evaluate_batch(batch, results)

            for problem, ev in zip(batch, evals):
                qid = problem.get("query_id", "")
                score = ev.get("score", 0.0)
                per_query_scores.setdefault(qid, []).append(score)
                trial_per_query.append({
                    "query_id": qid,
                    "score": score,
                    "prediction": ev.get("prediction", ""),
                    "expected_answer": ev.get("expected_answer", ""),
                })
                trial_acc_sum += score
                processed += 1

        trial_acc = trial_acc_sum / processed if processed else 0.0
        trial_detail = {
            "trial": trial,
            "forward_time_s": round(time.time() - trial_t0, 1),
            "per_query": trial_per_query,
            "accuracy": round(trial_acc, 4),
            "batch_size": batch_size,
        }
        all_trial_details.append(trial_detail)
        logger.info(f"Trial {trial} accuracy: {trial_acc:.4f}")

    total_time = time.time() - total_t0

    # ── 7. Compute aggregate metrics ─────────────────────────────────────────
    mean_n = _compute_mean_at_n(per_query_scores)
    pass_n = _compute_pass_at_n(per_query_scores)

    logger.info(f"=== Results (N={n_trials}) ===")
    logger.info(f"  mean@{n_trials}: {mean_n:.4f}")
    logger.info(f"  pass@{n_trials}: {pass_n:.4f}")
    logger.info(f"  Total time: {total_time:.1f}s")

    # ── 8. Save results ──────────────────────────────────────────────────────
    report = {
        "task_id": task_id,
        "mas_config_path": os.path.abspath(mas_config_path),
        "dataset_path": dataset_path,
        "split": split,
        "n_queries": len(dataset),
        "n_trials": n_trials,
        "total_time_s": round(total_time, 1),
        "metrics": {
            f"mean@{n_trials}": round(mean_n, 4),
            f"pass@{n_trials}": round(pass_n, 4),
        },
        "per_query_scores": {qid: [round(s, 4) for s in scores] for qid, scores in per_query_scores.items()},
        "trials": all_trial_details,
    }

    report_path = os.path.join(output_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(output_dir, "test_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(_build_summary_md(report))

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Summary saved to {summary_path}")

    return report


def _build_summary_md(report: Dict[str, Any]) -> str:
    n = report["n_trials"]
    metrics = report["metrics"]
    lines = [
        "# MAS Test Report",
        "",
        f"- **Task ID**: {report['task_id']}",
        f"- **MAS Config**: `{report['mas_config_path']}`",
        f"- **Dataset**: {report['dataset_path']} (split: {report['split']})",
        f"- **Queries**: {report['n_queries']}",
        f"- **Trials per query (N)**: {n}",
        f"- **Total time**: {report['total_time_s']}s",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| mean@{n} | {metrics[f'mean@{n}']:.4f} |",
        f"| pass@{n} | {metrics[f'pass@{n}']:.4f} |",
        "",
        "## Per-Trial Accuracy",
        "",
        "| Trial | Accuracy |",
        "|-------|----------|",
    ]
    for t in report["trials"]:
        lines.append(f"| {t['trial']} | {t['accuracy']:.4f} |")

    pqs = report.get("per_query_scores", {})
    lines.extend([
        "",
        "## Per-Query Breakdown",
        "",
        "| Query ID | Scores | mean | pass |",
        "|----------|--------|------|------|",
    ])
    for qid, scores in pqs.items():
        mean_s = sum(scores) / len(scores) if scores else 0.0
        pass_s = 1 if any(s > 0 for s in scores) else 0
        scores_str = ", ".join(f"{s:.1f}" for s in scores)
        lines.append(f"| {qid} | [{scores_str}] | {mean_s:.4f} | {pass_s} |")

    lines.append("")
    return "\n".join(lines)


def main():
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="OptiMAS Config Tester (pass@N / mean@N)")
    parser.add_argument(
        "--dataset", default=None,
        help="Path to a single JSONL dataset (ratio-split mode). "
             "Not required when --test-dataset is provided.",
    )
    parser.add_argument("--mas-config", required=True, help="Path to MAS config directory to evaluate")
    parser.add_argument("--task-id", default="test-run-001", help="Task ID (auto-generated if omitted)")
    parser.add_argument("--mcp-key", required=True, help="MCP API Key")
    parser.add_argument("--config-path", default="optimas_client/config/global_config.yaml")
    parser.add_argument("--n-trials", type=int, default=3, help="Number of trials per query (N)")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of queries to evaluate per forward batch")
    parser.add_argument("--max-mcp-concurrent", type=int, default=None)
    parser.add_argument("--split", default="test", choices=["test", "val", "train", "all"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--optimizer-config", default="optimas_client/config/optimizer_workbench/config.yaml")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=None,
                        help="Train/val/test split ratios (ratio-split mode only)")
    parser.add_argument("--seed", type=int, default=DEFAULT_DETERMINISTIC_SPLIT_SEED,
                        help=f"Random seed for train/val/test partitioning (default: {DEFAULT_DETERMINISTIC_SPLIT_SEED}).")
    parser.add_argument(
        "--train-dataset", default=None,
        help="Path to a pre-split JSONL file for training.",
    )
    parser.add_argument(
        "--val-dataset", default=None,
        help="Path to a pre-split JSONL file for validation.",
    )
    parser.add_argument(
        "--test-dataset", default=None,
        help="Path to a pre-split JSONL file for testing.",
    )
    args = parser.parse_args()

    config = ConfigLoader.load_config(config_path=args.config_path)
    workspace_root = config.get("workspace_base_dir", "./workspace")
    train_ts_suffix = _extract_train_timestamp_suffix_from_mas_config(
        args.mas_config,
        workspace_root,
    )
    if train_ts_suffix:
        suffix = train_ts_suffix
        logger.info(
            "Test workspace suffix matches training run under %s: %s",
            workspace_root,
            suffix,
        )
    else:
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("Test workspace suffix (no training run in path): %s", suffix)

    config_source_suffix = _extract_config_source_suffix(args.mas_config, workspace_root)
    task_id = f"{args.task_id}-test-{suffix}{config_source_suffix}"
    mcp_server_url = config.get("mcp", {}).get("server_url", "http://127.0.0.1:8188/mcp")

    asyncio.run(run_test(
        dataset_path=args.dataset,
        mas_config_path=args.mas_config,
        mcp_server_url=mcp_server_url,
        mcp_api_key=args.mcp_key,
        task_id=task_id,
        config_path=args.config_path,
        optimizer_config_path=args.optimizer_config,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        max_mcp_concurrent=args.max_mcp_concurrent,
        split=args.split,
        split_ratios=args.split_ratios,
        output_dir=args.output_dir,
        seed=args.seed,
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        test_dataset_path=args.test_dataset,
    ))


if __name__ == "__main__":
    main()
