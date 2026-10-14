"""Per-MCP-session WorkBench sandbox state (isolated DataFrame copies).

Each MCP session gets its own copy of the five domain DataFrames.  Before a
tool runs, the session copy is written into the cached module globals (under
``_tool_exec_lock``); after the tool returns, the globals are captured back.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from .runtime import WORKBENCH_ROOT, _tool_exec_lock, get_wb_modules


@dataclass
class WorkbenchSessionState:
    """Snapshot of all five WorkBench domain tables plus recorded actions."""

    calendar: pd.DataFrame
    email: pd.DataFrame
    analytics: pd.DataFrame
    project_management: pd.DataFrame
    customer_relationship_manager: pd.DataFrame
    function_calls: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Baseline (CSV reset) — caller must hold _tool_exec_lock
    # ------------------------------------------------------------------

    @classmethod
    def baseline(cls) -> "WorkbenchSessionState":
        """Re-read CSVs and return a pristine session state.

        ``reset_state()`` uses ``pd.read_csv`` with relative paths, so we
        ``chdir`` to ``WORKBENCH_ROOT`` while holding the tool-exec lock.
        No ``sys.modules`` swap — module refs are already cached.
        """
        mods = get_wb_modules()
        old_cwd = os.getcwd()
        os.chdir(WORKBENCH_ROOT.resolve())
        try:
            for mod in mods.values():
                mod.reset_state()
        finally:
            os.chdir(old_cwd)

        return cls(
            calendar=mods["calendar"].CALENDAR_EVENTS.copy(),
            email=mods["email"].EMAILS.copy(),
            analytics=mods["analytics"].PLOTS_DATA.copy(),
            project_management=mods["project_management"].PROJECT_TASKS.copy(),
            customer_relationship_manager=mods["customer_relationship_manager"].CRM_DATA.copy(),
        )

    # ------------------------------------------------------------------
    # Globals swap — caller must hold _tool_exec_lock
    # ------------------------------------------------------------------

    def apply_to_globals(self) -> None:
        """Write this session's DataFrames into the cached module globals."""
        mods = get_wb_modules()
        mods["calendar"].CALENDAR_EVENTS = self.calendar.copy()
        mods["email"].EMAILS = self.email.copy()
        mods["analytics"].PLOTS_DATA = self.analytics.copy()
        mods["project_management"].PROJECT_TASKS = self.project_management.copy()
        mods["customer_relationship_manager"].CRM_DATA = self.customer_relationship_manager.copy()

    def capture_from_globals(self) -> None:
        """Snapshot the cached module globals back into this session state."""
        mods = get_wb_modules()
        self.calendar = mods["calendar"].CALENDAR_EVENTS.copy()
        self.email = mods["email"].EMAILS.copy()
        self.analytics = mods["analytics"].PLOTS_DATA.copy()
        self.project_management = mods["project_management"].PROJECT_TASKS.copy()
        self.customer_relationship_manager = mods["customer_relationship_manager"].CRM_DATA.copy()


class WorkbenchSessionStore:
    """Thread-safe store keyed by MCP ``session_id``."""

    _lock = threading.Lock()
    _sessions: Dict[str, WorkbenchSessionState] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> WorkbenchSessionState:
        with cls._lock:
            state = cls._sessions.get(session_id)
            if state is None:
                with _tool_exec_lock:
                    state = WorkbenchSessionState.baseline()
                cls._sessions[session_id] = state
            return state

    @classmethod
    def reset(cls, session_id: str) -> WorkbenchSessionState:
        with cls._lock:
            with _tool_exec_lock:
                state = WorkbenchSessionState.baseline()
            cls._sessions[session_id] = state
            return state

    @classmethod
    def remove(cls, session_id: str) -> None:
        with cls._lock:
            cls._sessions.pop(session_id, None)

    @classmethod
    def get_recorded_actions(cls, session_id: str) -> List[str]:
        with cls._lock:
            state = cls._sessions.get(session_id)
            if state is None:
                return []
            return list(state.function_calls)
