"""Persist the optimizer's ``optimization_inputs`` alongside each training step.

:func:`write_optimizer_context_bundle` writes ``optimizer/optimizer_context_bundle.json``
into the step directory (``train_performance`` holds ``batch_report`` only), so a step's
full optimizer input remains recoverable from the run artifacts.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

BUNDLE_NAME = "optimizer_context_bundle.json"


def write_optimizer_context_bundle(
    step_dir: str, optimization_inputs: Dict[str, Any]
) -> None:
    """Write ``optimizer/optimizer_context_bundle.json`` under *step_dir*."""
    opt_dir = os.path.join(os.path.abspath(step_dir), "optimizer")
    os.makedirs(opt_dir, exist_ok=True)
    bundle = {"optimization_inputs": copy.deepcopy(optimization_inputs)}
    with open(os.path.join(opt_dir, BUNDLE_NAME), "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)


__all__ = ["BUNDLE_NAME", "write_optimizer_context_bundle"]
