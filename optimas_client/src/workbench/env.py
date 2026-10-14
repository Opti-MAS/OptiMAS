"""WorkBench runtime environment: path setup, cwd, and per-task state reset."""
from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

WORKBENCH_ROOT = Path(__file__).resolve().parents[3] / "optimas_server" / "workbench"

_workbench_path_added = False

# Serialise workbench_runtime(): os.chdir + reset_state use process-wide state.
_runtime_lock = threading.Lock()


def ensure_workbench_on_path() -> Path:
    """Add WorkBench repo root to ``sys.path`` once."""
    global _workbench_path_added
    root = WORKBENCH_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WorkBench root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _workbench_path_added = True
    return root


def reset_workbench_state() -> None:
    """Reset all sandbox DB modules to their CSV baselines."""
    from .official_metrics import _domains

    for domain in _domains():
        domain.reset_state()


@contextmanager
def workbench_runtime() -> Iterator[Path]:
    """Run with cwd at WorkBench root and fresh sandbox state.

    A lock serialises concurrent calls (from ``asyncio.to_thread``)
    because ``os.chdir`` and the domain globals are process-wide.
    """
    root = ensure_workbench_on_path()
    with _runtime_lock:
        old_cwd = os.getcwd()
        os.chdir(root)
        try:
            reset_workbench_state()
            yield root
        finally:
            from .official_metrics import _domains

            for domain in _domains():
                domain.reset_state()
            os.chdir(old_cwd)


def get_hardcoded_current_time():
    """Return WorkBench's fixed 'today' timestamp."""
    return datetime(2023, 11, 30)
