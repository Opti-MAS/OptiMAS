"""
Universal Evaluator for OptiMAS Client.

Architecture:
  UniversalEvaluator dispatches each (problem, result) pair to a registered
  EvaluationBackend based on query_type.  Backends are pluggable — add a new
  backend class, register it, and the evaluator routes automatically.

Built-in backends:
  - WorkBenchBackend (WorkBench official state-based action evaluation)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class EvaluationBackend(ABC):
    """Base class for evaluation backends."""

    name: str = "base"

    @abstractmethod
    async def evaluate(
        self,
        problem: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single (problem, result) pair.

        Must return a dict with at least ``score`` and ``reason`` keys.
        """


# ---------------------------------------------------------------------------
# Registry & Evaluator
# ---------------------------------------------------------------------------

from .workbench_eval import WorkBenchBackend  # noqa: E402

_DEFAULT_ROUTING: Dict[str, str] = {
    "workbench": "workbench",
}


class UniversalEvaluator:
    """Extensible evaluator with pluggable backends.

    Usage::

        evaluator = UniversalEvaluator(grader_llm=my_llm)
        # optionally register custom backends:
        evaluator.register_backend("my_type", MyBackend(...))
        results = await evaluator.evaluate_batch(problems, results)
    """

    def __init__(
        self,
        grader_llm: Any = None,
        routing: Optional[Dict[str, str]] = None,
        fallback_backend: str = "workbench",
    ):
        self.grader_llm = grader_llm
        self._routing = dict(_DEFAULT_ROUTING)
        if routing:
            self._routing.update(routing)
        self._fallback_backend = fallback_backend

        self._backends: Dict[str, EvaluationBackend] = {}
        self._backends["workbench"] = WorkBenchBackend()

    def register_backend(self, query_type: str, backend: EvaluationBackend) -> None:
        """Register a custom backend and route a query_type to it."""
        self._backends[backend.name] = backend
        self._routing[query_type] = backend.name

    def _resolve_backend(self, query_type: str) -> EvaluationBackend:
        backend_name = self._routing.get(query_type, self._fallback_backend)
        backend = self._backends.get(backend_name)
        if backend is None:
            logger.warning(
                "No backend '%s' for query_type '%s'; falling back to '%s'.",
                backend_name, query_type, self._fallback_backend,
            )
            backend = self._backends.get(self._fallback_backend)
        if backend is None:
            raise RuntimeError(
                f"Fallback backend '{self._fallback_backend}' not registered."
            )
        return backend

    async def evaluate_single(
        self, problem: Dict[str, Any], result: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_type = problem.get("query_type", "workbench")
        backend = self._resolve_backend(query_type)
        return await backend.evaluate(problem, result)

    async def evaluate_batch(
        self,
        problems: List[Dict[str, Any]],
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate many (problem, result) pairs concurrently; order is preserved."""
        if len(problems) != len(results):
            raise ValueError("Each problem must have exactly one execution result.")
        tasks = [
            self.evaluate_single(p, r)
            for p, r in zip(problems, results)
        ]
        return await asyncio.gather(*tasks)
