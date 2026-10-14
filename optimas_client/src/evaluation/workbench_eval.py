"""
WorkBench evaluation backend.

Uses the official WorkBench metrics from ``src.evals.utils``:
  - ``is_correct`` — state-based comparison after executing predicted vs gold actions
  - ``has_side_effects`` — unwanted state mutations
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from ..workbench.action_format import parse_ground_truth_actions
from ..workbench.env import workbench_runtime

logger = logging.getLogger(__name__)


def _extract_function_calls(result: Dict[str, Any]) -> List[str]:
    calls = result.get("function_calls")
    if isinstance(calls, list) and calls:
        return [str(c) for c in calls]
    data = result.get("data")
    if isinstance(data, dict) and isinstance(data.get("function_calls"), list):
        return [str(c) for c in data["function_calls"]]
    return []


def _evaluate_sync(
    predicted_actions: List[str],
    ground_truth_actions: List[str],
    error: str,
) -> Dict[str, Any]:
    from ..workbench.official_metrics import is_correct, has_side_effects, is_exact_match

    with workbench_runtime():
        exact = is_exact_match(predicted_actions, ground_truth_actions)
        correct = is_correct(predicted_actions, ground_truth_actions, error)
        side_effects = has_side_effects(predicted_actions, ground_truth_actions)

    return {
        "exact_match": exact,
        "correct": correct,
        "unwanted_side_effects": side_effects,
    }


class WorkBenchBackend:
    """Official WorkBench state-based evaluator."""

    name = "workbench"

    async def evaluate(
        self,
        problem: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        predicted_actions = _extract_function_calls(result)
        ground_truth_actions = parse_ground_truth_actions(
            problem.get("answer") or problem.get("ground_truth_actions")
        )
        agent_error = ""
        if not result.get("success"):
            agent_error = str(result.get("error") or "Agent execution failed")

        try:
            metrics = await asyncio.to_thread(
                _evaluate_sync,
                predicted_actions,
                ground_truth_actions,
                agent_error,
            )
        except Exception as exc:
            logger.error("WorkBench evaluation failed: %s", exc, exc_info=True)
            return {
                "problem_id": problem.get("query_id"),
                "score": 0.0,
                "prediction": str(predicted_actions),
                "extracted_answer": str(predicted_actions),
                "expected_answer": str(ground_truth_actions),
                "backend": self.name,
                "evaluation_reason": f"Evaluation error: {exc}",
                "correct": False,
                "exact_match": False,
                "unwanted_side_effects": False,
            }

        correct = metrics["correct"]
        reason_parts = [
            f"correct={correct}",
            f"exact_match={metrics['exact_match']}",
            f"unwanted_side_effects={metrics['unwanted_side_effects']}",
            f"predicted_actions={len(predicted_actions)}",
            f"ground_truth_actions={len(ground_truth_actions)}",
        ]
        if agent_error and not predicted_actions:
            reason_parts.append(f"agent_error={agent_error}")

        return {
            "problem_id": problem.get("query_id"),
            "score": 1.0 if correct else 0.0,
            "prediction": str(predicted_actions),
            "extracted_answer": str(predicted_actions),
            "expected_answer": str(ground_truth_actions),
            "backend": self.name,
            "evaluation_reason": "; ".join(reason_parts),
            "correct": correct,
            "exact_match": metrics["exact_match"],
            "unwanted_side_effects": metrics["unwanted_side_effects"],
            "predicted_actions": predicted_actions,
            "ground_truth_actions": ground_truth_actions,
        }
