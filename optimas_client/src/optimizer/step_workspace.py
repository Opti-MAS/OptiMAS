"""
Resolve per-sample MAS session folders under a train ``step_dir``.

MCP uses ``<query_id>_<session_suffix>/``; optimizers must use the real folder name, not
``query_id`` alone, when opening ``working_trajectory/`` or other artifacts.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


def find_sample_workspace_dir(step_dir: str, query_id: str) -> Optional[str]:
    """Return absolute path ``step_dir/<query_id>_<slug>/`` if that directory exists."""
    pattern = re.compile(rf"^{re.escape(str(query_id))}(?:-[0-9a-f]+)?_[0-9a-f]+$")
    try:
        for name in sorted(os.listdir(step_dir)):
            p = os.path.join(step_dir, name)
            if not os.path.isdir(p):
                continue
            if pattern.fullmatch(name):
                return p
    except OSError as e:
        logger.debug("listdir %s: %s", step_dir, e)
    return None


def sample_workspace_basename(step_dir: str, query_id: str) -> Optional[str]:
    """Directory name only (suitable as ``relpath`` from ``step_dir``)."""
    p = find_sample_workspace_dir(step_dir, query_id)
    return os.path.basename(p) if p else None
