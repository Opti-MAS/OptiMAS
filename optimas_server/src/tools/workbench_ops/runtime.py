"""WorkBench repo path setup; isolate ``src`` package from optimas_server.

Architecture
------------
*   Startup: ``workbench_cwd()`` swaps ``sys.modules`` + ``chdir`` once so that
    WorkBench's ``src.tools`` can be imported.  After importing, module object
    references are cached in ``_wb_modules`` via ``_cache_wb_modules()``.
*   Runtime: tool calls never touch ``sys.modules`` or ``chdir``.  They only
    swap per-session DataFrames into the cached module globals, protected by
    ``_tool_exec_lock``.  This eliminates the "No module named …" race and
    reduces lock-hold time to the DataFrame copy + pure-Python tool body.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

WORKBENCH_ROOT = Path(__file__).resolve().parents[3] / "workbench"

# ---------------------------------------------------------------------------
# Startup lock (reentrant): protects sys.modules swap during initial import.
# Used ONLY during server startup by workbench_cwd() / _load_tool_specs().
# ---------------------------------------------------------------------------
_startup_lock = threading.RLock()

# ---------------------------------------------------------------------------
# Tool execution lock: serialises  apply_globals → func → capture_globals.
# NOT reentrant — each tool call acquires exactly once.  No sys.modules swap
# and no chdir happen under this lock; only DataFrame copies + tool logic.
# ---------------------------------------------------------------------------
_tool_exec_lock = threading.Lock()

# Cached WorkBench domain module references (populated once at startup).
_wb_modules: Optional[Dict[str, Any]] = None


def _src_module_keys() -> list:
    return [k for k in sys.modules if k == "src" or k.startswith("src.")]


def ensure_workbench_on_path() -> Path:
    root = WORKBENCH_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"WorkBench root not found: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    return root


@contextmanager
def workbench_import_context() -> Iterator[Path]:
    """Swap ``sys.modules['src.*']`` to WorkBench for the duration of the block."""
    old_path = list(sys.path)
    optimas_src_snapshot: Dict[str, object] = {
        k: sys.modules[k] for k in _src_module_keys()
    }
    for k in optimas_src_snapshot:
        del sys.modules[k]

    root = ensure_workbench_on_path()
    try:
        yield root
    finally:
        wb_keys = _src_module_keys()
        for k in wb_keys:
            del sys.modules[k]
        sys.modules.update(optimas_src_snapshot)
        sys.path[:] = old_path


@contextmanager
def workbench_cwd() -> Iterator[Path]:
    """Full module-swap + chdir — used **only at startup** for initial imports.

    Runtime tool calls must use ``get_wb_modules()`` + ``_tool_exec_lock``.
    """
    with _startup_lock:
        with workbench_import_context() as root:
            old = os.getcwd()
            os.chdir(root)
            try:
                yield root
            finally:
                os.chdir(old)


# ------------------------------------------------------------------
# Module cache
# ------------------------------------------------------------------

def _cache_wb_modules() -> None:
    """Import and cache WorkBench domain modules.  **Must** be called inside
    ``workbench_cwd()`` so that ``src.tools.*`` resolves to WorkBench.
    """
    global _wb_modules
    from src.tools import (  # noqa: WPS433
        analytics,
        calendar,
        customer_relationship_manager,
        email,
        project_management,
    )
    _wb_modules = {
        "calendar": calendar,
        "email": email,
        "analytics": analytics,
        "project_management": project_management,
        "customer_relationship_manager": customer_relationship_manager,
    }


def get_wb_modules() -> Dict[str, Any]:
    """Return cached WorkBench module references, importing on first call."""
    global _wb_modules
    if _wb_modules is not None:
        return _wb_modules
    with workbench_cwd():
        _cache_wb_modules()
    return _wb_modules  # type: ignore[return-value]
