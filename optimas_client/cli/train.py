"""
Training Loop for MAS Evolution

Orchestrates: DataLoader -> MAS Forward -> Evaluator -> Optimizer -> Checkpoint
Supports ``--resume-from`` to continue training from a validated checkpoint.
"""

import asyncio
import copy
import logging
import math
import random
import time
import os
import shutil
from typing import Dict, Any, List, Optional

from ..src.data.dataloader import (
    UniversalDataLoader,
    AdaptiveSampler,
    DEFAULT_DETERMINISTIC_SPLIT_SEED,
    DEFAULT_TRAIN_VAL_TEST_RATIOS,
)
from ..src.evaluation.evaluator import UniversalEvaluator
from ..src.engine.forward_engine import MASForwardEngine
from ..src.llm.base import configure_global_llm_concurrency
from ..src.optimizer.optimizer_agent import OptimizerAgent
from ..src.optimizer.batch_trajectory_report import build_batch_report
from ..src.optimizer.optimizer_context_restore import write_optimizer_context_bundle
from ..src.mcp.client import MCPClient
from ..src.config.loader import ConfigLoader
from ..src.config.mas_config_loader import get_entry_agent_name_from_config
from ..src.llm.factory import create_llm
from ..src.messaging.emitter import MessageEmitter

from .train_utils import (
    compute_accuracy,
    build_aggregated_stats,
    save_metrics,
    append_optimization_history,
    archive_optimizer_working_log,
    carry_forward_optimizer_memory,
    parse_resume_checkpoint,
    restore_prev_state,
    cleanup_steps_after,
)

logger = logging.getLogger(__name__)


async def run_training(
    dataset_path: Optional[str] = None,
    mcp_server_url: str = "",
    mcp_api_key: str = "",
    task_id: str = "",
    config_path: str = "optimas_client/config/global_config.yaml",
    optimizer_config_path: str = "optimas_client/config/optimizer_workbench/config.yaml",
    max_epochs: int = 10,
    batch_size: int = 10,
    max_val_batch_size: Optional[int] = None,
    max_mcp_concurrent: Optional[int] = None,
    split_ratios: Optional[List[float]] = None,
    resume_from: Optional[str] = None,
    retry_ratio: float = 0.5,
    boost_factor: float = 1.2,
    decay_factor: float = 0.6,
    reset_interval: Optional[int] = None,
    train_dataset_path: Optional[str] = None,
    val_dataset_path: Optional[str] = None,
    test_dataset_path: Optional[str] = None,
    seed: int = DEFAULT_DETERMINISTIC_SPLIT_SEED,
):
    """Main training loop with adaptive sampling.

    Uses ``AdaptiveSampler`` for probability-weighted batch construction:
    failed queries from the previous step are retried so the optimizer can
    immediately verify its fixes, while fresh queries maintain coverage.
    """
    logger.info("Starting MAS Training Loop...")
    if split_ratios is None:
        split_ratios = list(DEFAULT_TRAIN_VAL_TEST_RATIOS)

    global_config = ConfigLoader.load_config(config_path=config_path)
    optimizer_config_data = ConfigLoader.load_config(config_path=optimizer_config_path)
    global_config.update(optimizer_config_data)

    optimizer_config = global_config.get("optimizer", {})
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if max_val_batch_size is not None and max_val_batch_size < 1:
        raise ValueError("max_val_batch_size must be positive when supplied.")
    if not dataset_path and not (train_dataset_path or val_dataset_path or test_dataset_path):
        benchmark = optimizer_config.get("benchmark", {})
        train_dataset_path = benchmark.get("train_dataset_file_path")
        val_dataset_path = benchmark.get("val_dataset_file_path")
    if not task_id or os.path.basename(task_id) != task_id or task_id in (".", ".."):
        raise ValueError("task_id must be a single safe workspace directory name.")

    # ── DataLoader ───────────────────────────────────────────────────────────
    use_presplit = bool(train_dataset_path or val_dataset_path or test_dataset_path)
    if use_presplit:
        dataloader = UniversalDataLoader(
            seed=seed,
            train_path=train_dataset_path,
            val_path=val_dataset_path,
            # The held-out test file is never read by training.
        )
    else:
        if dataset_path is None:
            raise ValueError(
                "Either --dataset or --train-dataset/--val-dataset must be provided."
            )
        dataloader = UniversalDataLoader(
            dataset_path=dataset_path, seed=seed,
            train_ratio=split_ratios[0], val_ratio=split_ratios[1], test_ratio=split_ratios[2],
        )
    n_train, n_val, n_test = len(dataloader.train_set), len(dataloader.val_set), len(dataloader.test_set)
    if not n_val:
        raise ValueError("A non-empty validation split is required for training.")
    logger.info(f"Dataset split (seed={seed}): total={n_train+n_val+n_test} | train={n_train} | val={n_val} | test={n_test}")

    # ── Adaptive sampler ─────────────────────────────────────────────────────
    sampler = AdaptiveSampler(
        dataloader,
        batch_size,
        retry_ratio=retry_ratio,
        boost_factor=boost_factor,
        decay_factor=decay_factor,
        reset_interval=reset_interval,
        seed=seed,
    )

    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * max_epochs
    logger.info(
        f"Adaptive sampler: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
        f"retry_ratio={retry_ratio}, boost={boost_factor}, decay={decay_factor}, "
        f"reset_interval={sampler.reset_interval}"
    )

    grader_llm_config = optimizer_config.get("benchmark", {}).get("grader_llm")
    grader_llm = None
    if grader_llm_config:
        grader_llm = create_llm(
            grader_llm_config,
            ConfigLoader.get_api_recording_config(global_config),
            ConfigLoader.get_llm_settings(global_config),
        )
    evaluator = UniversalEvaluator(grader_llm=grader_llm)

    llm_concurrency = global_config.get("max_llm_concurrency", batch_size * 4)
    configure_global_llm_concurrency(llm_concurrency)

    workspace_root = global_config.get("workspace_base_dir", "./workspace")
    mcp_concurrent = max_mcp_concurrent if max_mcp_concurrent is not None else batch_size
    per_problem_timeout = global_config.get("per_problem_timeout")
    engine = MASForwardEngine(
        mcp_server_url=mcp_server_url,
        mcp_api_key=mcp_api_key,
        workspace_root=workspace_root,
        max_concurrent=mcp_concurrent,
        per_problem_timeout=per_problem_timeout,
    )

    config_source = optimizer_config.get("config_source")
    if not config_source:
        raise ValueError("optimizer.config_source is required in config.yaml")

    opt_agent_config = ConfigLoader.get_agent_config(global_config, "optimizer")
    opt_agent_config["_config_dir"] = global_config.get("_config_dir", ".")
    messaging_config = global_config.get("messaging", {})

    # ── State variables (may be overridden by resume) ────────────────────────
    prev_val_acc = 0.0
    prev_optimization_summary = "Initial state, no previous optimization."
    step_counter = 0
    best_val_acc = -1.0
    best_val_config_dir: Optional[str] = None
    prev_opt_dir: Optional[str] = None
    current_mas_config_dir = os.path.abspath(config_source)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if resume_from:
        logger.info("=== Resuming training from checkpoint: %s ===", resume_from)
        ckpt = parse_resume_checkpoint(resume_from)
        workspace_dir = ckpt["workspace_dir"]
        task_id = os.path.basename(workspace_dir)

        restored = restore_prev_state(ckpt)
        step_counter = ckpt["step"]
        current_mas_config_dir = os.path.abspath(ckpt["mas_config_dir"])
        prev_opt_dir = ckpt["opt_dir"]
        prev_val_acc = restored["prev_val_acc"]
        prev_optimization_summary = restored["prev_optimization_summary"]
        best_val_acc = restored["best_val_acc"]
        best_val_config_dir = restored["best_val_config_dir"]

        run_metrics: Dict[str, Any]
        if restored["run_metrics"] is not None:
            run_metrics = restored["run_metrics"]
        else:
            run_metrics = {
                "task_id": task_id,
                "dataset_path": dataset_path,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "split": {"total": n_train + n_val + n_test, "train": n_train, "val": n_val, "test": n_test},
                "steps": [],
                "best_val": None,
            }

        # Restore adaptive sampler state
        if sampler.load_state(ckpt["opt_dir"]):
            logger.info("Adaptive sampler state restored from checkpoint.")
        else:
            raise ValueError("Resume checkpoint has no matching sampler state.")

        cleanup_steps_after(workspace_dir, ckpt["epoch"], ckpt["step"])

        logger.info(
            "Resumed state: step=%d, prev_val_acc=%.4f, best_val_acc=%.4f",
            step_counter, prev_val_acc, best_val_acc,
        )
    else:
        workspace_dir = os.path.join(workspace_root, task_id)
        os.makedirs(workspace_dir, exist_ok=False)
        _dataset_info: Dict[str, Any] = {"dataset_path": dataset_path}
        if use_presplit:
            _dataset_info.update({
                "train_dataset_path": train_dataset_path,
                "val_dataset_path": val_dataset_path,
                "test_dataset_path": test_dataset_path,
            })
        run_metrics = {
            "task_id": task_id,
            **_dataset_info,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "split": {"total": n_train + n_val + n_test, "train": n_train, "val": n_val, "test": n_test},
            "steps": [],
            "best_val": None,
        }

    last_valid_mas_config_dir = current_mas_config_dir
    val_set = dataloader.get_validation_set()
    # An explicit lightweight cap still selects the same validation queries every step.
    val_batch = (random.Random(seed).sample(val_set, max_val_batch_size)
                 if max_val_batch_size is not None and len(val_set) > max_val_batch_size else val_set)

    # ── Main training loop (step-count based) ────────────────────────────────
    remaining_steps = total_steps - step_counter
    for _ in range(remaining_steps):
        step_counter += 1
        epoch = (step_counter - 1) // steps_per_epoch
        logger.info(f"--- Step {step_counter}/{total_steps} (epoch {epoch}) ---")
        try:
            train_batch, resampled_ids = sampler.get_batch()
            logger.info(
                f"Sampled {len(train_batch)} training examples "
                f"({len(resampled_ids)} resampled from prev failures)."
            )

            step_rel = f"epoch_{epoch}/step_{step_counter}"
            step_dir = os.path.join(workspace_dir, step_rel)
            os.makedirs(step_dir, exist_ok=True)

            step_mas_config_dir = os.path.join(step_dir, "mas_config")
            if os.path.abspath(current_mas_config_dir) != os.path.abspath(step_mas_config_dir):
                shutil.copytree(current_mas_config_dir, step_mas_config_dir)

            step_mas_config = ConfigLoader.load_config(
                step_mas_config_dir,
            )

            opt_dir = os.path.join(step_dir, "optimizer")
            os.makedirs(opt_dir, exist_ok=True)
            with open(os.path.join(opt_dir, "previous_summary.md"), "w", encoding="utf-8") as f:
                f.write(prev_optimization_summary)

            if prev_opt_dir and os.path.isdir(prev_opt_dir):
                carry_forward_optimizer_memory(prev_opt_dir, opt_dir)

            train_ws = f"{task_id}/{step_rel}"
            val_ws = f"{task_id}/epoch_{epoch}/step_{step_counter}_validation"
            evaluated_config_dir = os.path.join(workspace_dir, f"{step_rel}_validation", "mas_config")
            shutil.copytree(step_mas_config_dir, evaluated_config_dir)

            logger.info("Forward pass (train + validation)...")
            t0 = time.time()
            train_results, val_results = await asyncio.gather(
                engine.forward(train_batch, step_mas_config, workspace_subdir=train_ws, global_config=global_config),
                engine.forward(val_batch, step_mas_config, workspace_subdir=val_ws, global_config=global_config),
            )
            logger.info(f"Forward pass done in {time.time()-t0:.1f}s")

            train_evals, val_evals = await asyncio.gather(
                evaluator.evaluate_batch(train_batch, train_results),
                evaluator.evaluate_batch(val_batch, val_results),
            )

            train_acc = compute_accuracy(train_evals)
            current_val_acc = compute_accuracy(val_evals)
            val_diff = current_val_acc - prev_val_acc
            logger.info(f"Train acc: {train_acc:.4f} | Val acc: {current_val_acc:.4f} (diff: {val_diff:+.4f})")

            # ── Update adaptive sampler weights ──────────────────────────
            weight_updates = []
            for problem, ev in zip(train_batch, train_evals):
                qid = str(
                    ev.get("problem_id")
                    or problem.get("query_id")
                    or problem.get("id")
                    or problem.get("instance_id")
                    or ""
                )
                is_correct = ev.get("score", 0) > 0
                weight_updates.append((qid, is_correct))
            sampler.update_weights(weight_updates)

            train_stats = build_aggregated_stats(train_evals)
            val_stats = build_aggregated_stats(val_evals)

            entry_name = get_entry_agent_name_from_config(step_mas_config)
            all_agent_names = sorted(step_mas_config.get("agents", {}).keys())
            try:
                train_batch_report = build_batch_report(
                    train_batch,
                    train_evals,
                    train_results,
                    train_stats,
                    entry_agent_name=entry_name,
                    step_dir=step_dir,
                    configured_agents=all_agent_names,
                    resampled_query_ids=resampled_ids,
                )
            except Exception as br_err:
                logger.warning("build_batch_report failed, optimizer will use fallback: %s", br_err)
                train_batch_report = None

            logger.info("Optimizer backward pass...")

            optimization_inputs = {
                "train_performance": {
                    "batch_report": train_batch_report,
                },
                "validation_signal": {
                    "current_acc": current_val_acc,
                    "acc_diff": val_diff,
                },
                "previous_optimization": prev_optimization_summary,
                "step_dir": step_dir,
                "resampled_query_ids": resampled_ids,
                "progress": {
                    "current_epoch": epoch,
                    "total_epochs": max_epochs,
                    "current_step": step_counter,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                },
            }
            try:
                write_optimizer_context_bundle(step_dir, optimization_inputs)
            except Exception as snap_e:
                logger.warning("write_optimizer_context_bundle: %s", snap_e)

            opt_ws = f"{task_id}/{step_rel}"
            async with MCPClient(mcp_server_url, mcp_api_key, "optimizer", workspace_subdir=opt_ws, skip_session_dir=True) as opt_mcp:
                opt_traj_base = os.path.join(opt_dir, "working_trajectory")
                os.makedirs(opt_traj_base, exist_ok=True)

                opt_msg_config = copy.deepcopy(messaging_config)
                handlers = opt_msg_config.get("handlers") or []
                fh = next((h for h in handlers if isinstance(h, dict) and h.get("type") == "file"), None)
                if fh is not None:
                    fh["base_path"] = opt_traj_base
                else:
                    opt_msg_config.setdefault("handlers", []).append({"type": "file", "base_path": opt_traj_base})

                optimizer = OptimizerAgent(
                    mcp_client=opt_mcp,
                    config=opt_agent_config,
                    message_emitter=MessageEmitter.from_config(
                        opt_msg_config,
                        context={"task_id": task_id, "step": step_counter, "agent_name": "optimizer",
                            "invocation_id": os.urandom(8).hex()},
                    ),
                    recording_config=ConfigLoader.get_api_recording_config(global_config),
                    global_llm_settings=ConfigLoader.get_llm_settings(global_config),
                    global_config=global_config,
                )

                t0 = time.time()
                step_mas_config, prev_optimization_summary = await optimizer.step(optimization_inputs, step_mas_config)
                logger.info(f"Backward pass done in {time.time()-t0:.1f}s")

            with open(os.path.join(opt_dir, "optimization_details.md"), "w", encoding="utf-8") as f:
                f.write(prev_optimization_summary)

            _batch_agg = (train_batch_report or {}).get("batch_aggregates", {})
            append_optimization_history(
                opt_dir, step_counter, epoch,
                train_acc, current_val_acc, val_diff,
                len(train_batch), _batch_agg, val_stats,
                prev_optimization_summary,
            )

            archive_optimizer_working_log(opt_dir, step_counter)
            prev_opt_dir = opt_dir
            current_mas_config_dir = os.path.abspath(step_mas_config_dir)
            last_valid_mas_config_dir = current_mas_config_dir

            prev_val_acc = current_val_acc
            if current_val_acc >= best_val_acc:
                best_val_acc = current_val_acc
                best_val_config_dir = evaluated_config_dir
                logger.info(f"New best val acc: {best_val_acc:.4f} at step {step_counter}")

            step_row: Dict[str, Any] = {
                "epoch": epoch,
                "step": step_counter,
                "train_acc": round(train_acc, 4),
                "val_acc": round(current_val_acc, 4),
                "val_acc_diff": round(val_diff, 4),
                "resampled_count": len(resampled_ids),
                "evaluated_config_dir": os.path.abspath(evaluated_config_dir),
            }
            run_metrics["steps"].append(step_row)
            save_metrics(run_metrics, workspace_dir)

            # Each completed step has its own sampler checkpoint.
            sampler.save_state(opt_dir)
            sampler.save_state(workspace_dir)

            logger.info(f"--- Step {step_counter} done ---\n")

        except Exception as step_err:
            logger.error(f"Step {step_counter} failed: {step_err}", exc_info=True)
            run_metrics["steps"].append({
                "epoch": epoch,
                "step": step_counter,
                "error": str(step_err),
            })
            save_metrics(run_metrics, workspace_dir)
            raise RuntimeError(f"Training stopped at incomplete step {step_counter}; no unfinalized configuration is exported.") from step_err

    logger.info("=== Post-training: saving MAS configs ===")

    last_config_dst = os.path.join(workspace_dir, "last_mas_config")
    if os.path.isdir(last_config_dst):
        shutil.rmtree(last_config_dst)
    shutil.copytree(last_valid_mas_config_dir, last_config_dst)
    logger.info(f"Last MAS config saved to {last_config_dst}")

    if best_val_config_dir is not None:
        best_config_dst = os.path.join(workspace_dir, "best_mas_config")
        if os.path.isdir(best_config_dst):
            shutil.rmtree(best_config_dst)
        shutil.copytree(best_val_config_dir, best_config_dst)
        logger.info(f"Best MAS config (val_acc={best_val_acc:.4f}) saved to {best_config_dst}")
        run_metrics["best_val"] = {"val_acc": round(best_val_acc, 4), "config_dir": best_config_dst}
    else:
        logger.warning("No best val config recorded (all steps may have failed).")
        run_metrics["best_val"] = {"val_acc": None, "config_dir": None}

    save_metrics(run_metrics, workspace_dir)
    logger.info("Training finished.")


def main():
    import argparse
    import datetime

    parser = argparse.ArgumentParser(description="OptiMAS Training Loop")
    parser.add_argument(
        "--dataset", default=None,
        help="Path to a single JSONL dataset (ratio-split mode). "
             "Not required when --train-dataset is provided.",
    )
    parser.add_argument("--task-id", required=True, help="Task ID / workspace name")
    parser.add_argument("--mcp-key", required=True, help="MCP API Key")
    parser.add_argument("--config-path", default="optimas_client/config/global_config.yaml")
    parser.add_argument("--optimizer-config", default="optimas_client/config/optimizer_workbench/config.yaml")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Maximum optimization horizon in epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Training batch capacity per step (default: 10).")
    parser.add_argument("--seed", type=int, default=DEFAULT_DETERMINISTIC_SPLIT_SEED,
                        help=f"Random seed for train/val/test partitioning (default: {DEFAULT_DETERMINISTIC_SPLIT_SEED}).")
    parser.add_argument("--max-val-batch-size", type=int, default=None,
                        help="Optional fixed subset for lightweight runs; default evaluates the entire validation split.")
    parser.add_argument("--max-mcp-concurrent", type=int, default=None)
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_TRAIN_VAL_TEST_RATIOS),
        help="Train / val / test split proportions (ratio-split mode only).",
    )
    parser.add_argument(
        "--train-dataset", default=None,
        help="Path to a pre-split JSONL file for training. "
             "When provided, --split-ratios is ignored.",
    )
    parser.add_argument(
        "--val-dataset", default=None,
        help="Path to a pre-split JSONL file for validation.",
    )
    parser.add_argument(
        "--test-dataset", default=None,
        help="Path to a pre-split JSONL file for testing.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to the last valid step directory to resume from. "
            "Example: workspace/workbench-train-2026.../epoch_2/step_18. "
            "All steps after this one will be deleted and training continues."
        ),
    )
    # Adaptive sampler hyperparameters
    parser.add_argument("--retry-ratio", type=float, default=0.5,
                        help="Fraction of batch reserved for retrying previous failures (0-1).")
    parser.add_argument("--boost-factor", type=float, default=1.2,
                        help="Weight multiplier for failed queries.")
    parser.add_argument("--decay-factor", type=float, default=0.6,
                        help="Weight multiplier for correct queries.")
    parser.add_argument("--reset-interval", type=int, default=None,
                        help="Steps between full weight resets (default: 2 * ceil(N/B)).")
    args = parser.parse_args()

    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"{args.task_id}-train-{suffix}"

    config = ConfigLoader.load_config(config_path=args.config_path)
    mcp_server_url = config.get("mcp", {}).get("server_url", "http://127.0.0.1:8188/mcp")

    asyncio.run(run_training(
        dataset_path=args.dataset,
        mcp_server_url=mcp_server_url,
        mcp_api_key=args.mcp_key,
        task_id=task_id,
        config_path=args.config_path,
        optimizer_config_path=args.optimizer_config,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        max_val_batch_size=args.max_val_batch_size,
        max_mcp_concurrent=args.max_mcp_concurrent,
        split_ratios=args.split_ratios,
        resume_from=args.resume_from,
        retry_ratio=args.retry_ratio,
        boost_factor=args.boost_factor,
        decay_factor=args.decay_factor,
        reset_interval=args.reset_interval,
        seed=args.seed,
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        test_dataset_path=args.test_dataset,
    ))


if __name__ == "__main__":
    main()
