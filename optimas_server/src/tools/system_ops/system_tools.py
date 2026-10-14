"""
System-level workspace tools (tree / delete / move) built on top of session workspace.
"""
from __future__ import annotations
from pathlib import Path
import shutil
from typing import Annotated

from mcp import McpError
from mcp.types import ErrorData
from pydantic import Field

from src.config import get_settings
from src.tools.common_utils.path_utils import get_session_workspace, resolve_workspace_path, get_client_path
from src.utils.workspace_visibility import should_hide_session_private_list_dir
from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR


def format_paths_as_tree(
    root_label: str, entries: list[Path], base: Path, max_depth: int
) -> str:
    """Render a simple tree from paths under *base*.

    *entries* are absolute or normalized paths. Lexical `relative_to` is used first so
    symlink directories whose targets lie outside the workspace still appear in the tree
    (when the symlink path is under *base*).
    """
    tree: dict = {}
    base_resolved = base.resolve()

    for path in entries:
        try:
            try:
                rel = path.relative_to(base_resolved)
            except ValueError:
                rel = path.resolve().relative_to(base_resolved)
        except Exception:
            continue
        parts = rel.parts
        if not parts:
            continue
        node = tree
        for idx, part in enumerate(parts):
            is_last = idx == len(parts) - 1
            if part not in node:
                node[part] = {
                    "_is_dir": path.is_dir() if is_last else True,
                    "_children": {},
                }
            if not is_last:
                node = node[part]["_children"]

    lines: list[str] = [root_label]

    def walk(subtree: dict, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        items = sorted(subtree.items(), key=lambda kv: (not kv[1]["_is_dir"], kv[0]))
        for idx, (name, meta) in enumerate(items):
            is_last = idx == len(items) - 1
            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{name}"
            if meta["_is_dir"]:
                line = f"{line}/"
            lines.append(line)
            if meta["_is_dir"] and meta["_children"] and depth < max_depth:
                extension = "    " if is_last else "│   "
                walk(meta["_children"], prefix + extension, depth + 1)

    walk(tree, prefix="", depth=1)
    return "\n".join(lines)

def _register_system_tools(mcp):
    """Register system-level tools (list_file_tree / delete_file / move_file)."""

    @mcp.tool
    async def list_file_tree(
        path: Annotated[str, Field(description="Optional path (relative to workspace root). If no provided, list the whole workspace.")] = "",
        max_depth: Annotated[int, Field(description="Maximum tree depth relative to the provided path (1 = only direct children of the path).", ge=1, le=20)] = 5,
        follow_links: Annotated[
            bool,
            Field(
                description=(
                    "If true, traverse into symlinked directories. If false, symlinked directories "
                    "still appear in the tree (so you can see e.g. per-query `working_trajectory/`) "
                    "but their contents are not expanded."
                ),
            ),
        ] = False,
    ) -> dict:
        """
        List files and directories under the session workspace as a tree string.
        """
        try:
            workspace = await get_session_workspace()
            if path:
                base = await resolve_workspace_path(path)
            else:
                base = workspace

            if not base.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path does not exist: {path or '/'}",
                    )
                )

            # Collect entries under base (including base if it's a file)
            entries: list[Path] = []
            format_base = base  # Base path for depth calculation in `format_paths_as_tree`
            if base.is_file():
                entries.append(base)
                root_label = "/" + await get_client_path(base.resolve())
                # For files, use parent directory as base for depth calculation
                format_base = base.parent
            else:
                root_label = "/"
                if base != workspace:
                    root_label = f"/{await get_client_path(base.resolve())}"

                # Get ignore paths from config
                settings = get_settings()
                ignore_paths = set(settings.workspace.ignore_paths)
                hide_wt = bool(
                    getattr(settings.workspace, "hide_own_working_trajectory_in_list", False)
                )

                # Walk directory tree up to a generous depth; formatting will clamp to max_depth.
                # When follow_links is false: still *list* symlinked dirs (e.g. `working_trajectory`
                # pointing at storage) so agents see the name, but do not descend into the target.
                def walk_dir(current: Path, current_depth: int) -> None:
                    if current_depth > max_depth:
                        return
                    try:
                        for entry in current.iterdir():
                            # Skip paths in ignore list
                            if entry.name in ignore_paths:
                                continue
                            try:
                                is_dir = entry.is_dir()
                                is_symlink = entry.is_symlink()
                                if not entry.resolve().is_relative_to(workspace.resolve()):
                                    continue
                            except OSError:
                                continue
                            if hide_wt and is_dir and should_hide_session_private_list_dir(
                                entry, workspace
                            ):
                                continue
                            entries.append(entry)
                            if is_dir:
                                if is_symlink and not follow_links:
                                    continue
                                walk_dir(entry, current_depth + 1)
                    except OSError as e:
                        # Surface a clear error rather than silently skipping
                        raise McpError(
                            ErrorData(
                                code=INTERNAL_ERROR,
                                message=f"Failed to list directory '{current}': {e}",
                            )
                        ) from e

                walk_dir(base, current_depth=1)

            tree = format_paths_as_tree(
                root_label, entries, format_base, max_depth
            )
            return {"success": True, "data": tree}

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"list_file_tree failed: {e}",
                )
            ) from e

    @mcp.tool
    async def delete_file(
        path: Annotated[str, Field(description="File or directory path relative to the workspace root")],
    ) -> dict:
        """
        Delete a file or directory (recursively) under the session workspace.
        """
        try:
            target = await resolve_workspace_path(path)

            if not target.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path does not exist: {path}",
                    )
                )

            if target.is_dir():
                try:
                    shutil.rmtree(target)
                except Exception as e:
                    raise McpError(
                        ErrorData(
                            code=INTERNAL_ERROR,
                            message=f"Failed to delete directory '{path}': {e}",
                        )
                    ) from e
                deleted_type = "directory"
            else:
                try:
                    target.unlink()
                except Exception as e:
                    raise McpError(
                        ErrorData(
                            code=INTERNAL_ERROR,
                            message=f"Failed to delete file '{path}': {e}",
                        )
                    ) from e
                deleted_type = "file"

            return {
                "success": True,
                "data": f"Successfully deleted {deleted_type}: {path}",
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"delete_file failed: {e}",
                )
            ) from e

    @mcp.tool
    async def move_file(
        old_path: Annotated[str, Field(description="Source path relative to workspace root")],
        new_path: Annotated[str, Field(description="Destination path relative to workspace root")],
        keep_old: Annotated[bool, Field(description="If True, copy; if False, move/rename.")] = False,
    ) -> dict:
        """
        Copy or move a file within the session workspace.
        """
        try:
            src = await resolve_workspace_path(old_path)
            dst = await resolve_workspace_path(new_path)

            if not src.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Source path does not exist: {old_path}",
                    )
                )

            if not src.is_file():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Source path is not a file: {old_path}",
                    )
                )

            if not keep_old and dst.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Destination already exists: {new_path}",
                    )
                )

            dst.parent.mkdir(parents=True, exist_ok=True)

            if keep_old:
                shutil.copy2(src, dst)
                operation = "copied"
            else:
                src.rename(dst)
                operation = "moved"

            return {
                "success": True,
                "data": f"Successfully {operation} {old_path} to {new_path}",
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"move_file failed: {e}",
                )
            ) from e
