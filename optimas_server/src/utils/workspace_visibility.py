"""Rules for hiding session-local paths from list_file_tree (no FastMCP imports)."""

from __future__ import annotations

from pathlib import Path

# Directory names hidden at session root (and under optimizer/) when hide flag is on.
_PRIVATE_LIST_DIR_NAMES = frozenset({"working_trajectory", "tool_call"})


def should_hide_session_private_list_dir(entry: Path, session_workspace: Path) -> bool:
    """Return True if *entry* is this MCP session's own private log dir (omit from list_file_tree).

    Hides only, relative to *session_workspace*:

    - ``working_trajectory`` or ``tool_call`` as a **direct** child (MAS worker session root)
    - ``optimizer/working_trajectory`` or ``optimizer/tool_call`` (optimizer logs under a train step)

    Does **not** hide deeper paths such as ``<query_id>/working_trajectory`` or
    ``<query_id>/tool_call`` so an optimizer session rooted at ``step_x`` can still list
    per-query worker trajectories and tool-call logs.
    """
    try:
        if entry.name not in _PRIVATE_LIST_DIR_NAMES:
            return False
        if not entry.is_dir():
            return False
        ws = session_workspace.resolve()
        # Prefer lexical relative path so symlinked per-query dirs are
        # classified by their visible location in the session tree, not by
        # resolved target.
        try:
            rel = entry.relative_to(ws)
        except ValueError:
            rel = entry.resolve().relative_to(ws)
        parts = rel.parts
        if len(parts) == 1 and parts[0] in _PRIVATE_LIST_DIR_NAMES:
            return True
        if (
            len(parts) == 2
            and parts[0] == "optimizer"
            and parts[1] in _PRIVATE_LIST_DIR_NAMES
        ):
            return True
        return False
    except (ValueError, OSError):
        return False


