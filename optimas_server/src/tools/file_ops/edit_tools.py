"""
File edit operations tools.

Provides:
- patch_file: Exact substring find-and-replace (optional fuzzy hint on miss); returns unified diff
- file_edit: Edit file by replacing old_lines with new_lines using context matching
- replace_file_block: Replace file content in a specific line range
"""
import asyncio
import difflib
import os
import re
from pathlib import Path
from typing import Annotated, List, Optional, Tuple

import aiofiles
from mcp import McpError
from mcp.types import ErrorData
from pydantic import Field

from src.tools.common_utils.path_utils import get_session_workspace, resolve_workspace_path
from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR

_ENCODING_UTF8 = "utf-8"
_FUZZY_SUGGEST_MAX_LINES = 3

# Matches read_file line-number gutters at the start of a line.
# Two distinct branches via alternation — each consumes only the gutter
# (including separator whitespace) so the remaining text == actual content.
#
#   New format  "   1 | content"  →  branch 1, group(1)="1",  remaining="content"
#   Old format  "1|content"       →  branch 2, group(2)="1",  remaining="content"
#
# Branch 1: \s*(\d+) \| <space>  — the " | " is a fixed literal in the new format
# Branch 2: (\d+)\|              — digit(s) glued to pipe, no surrounding spaces
_LINE_NUMBER_PREFIX_RE = re.compile(r"^(?:\s*(\d+) \| |(\d+)\|)")


def _extract_line_number(m: re.Match) -> int:
    """Return the line number captured by ``_LINE_NUMBER_PREFIX_RE``."""
    return int(m.group(1) or m.group(2))


async def _try_repair_docker_owned_path(path: Path) -> None:
    """Best-effort repair for root-owned files created through Docker bind mounts."""
    try:
        from fastmcp.server.dependencies import get_context
        from src.middleware.mcp_request_context import get_mcp_tool_session_id_any
        from src.tools.common_utils.async_state import get_state_safe
        from src.tools.common_utils.path_utils import _session_middleware

        session_id = get_mcp_tool_session_id_any()
        if not session_id:
            ctx = get_context()
            if ctx is not None:
                session_id = await get_state_safe(ctx, "session_id")
        if not session_id or not _session_middleware:
            return

        session = _session_middleware._sessions.get(str(session_id).strip())
        container_id = (
            session.metadata.get("docker_container_id")
            if session and session.metadata else None
        )
        if not container_id:
            return

        workspace = await get_session_workspace()
        rel = path.resolve().relative_to(workspace.resolve())
        docker_path = "/testbed" if rel.as_posix() == "." else f"/testbed/{rel.as_posix()}"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_id,
            "chown", f"{os.getuid()}:{os.getgid()}", docker_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)
    except Exception:
        return


def _strip_line_number_prefixes(text: str) -> Optional[str]:
    """Strip ``read_file`` line-number gutters from *text* if present.

    Safety checks (all must pass):
    1. **Every** non-empty line starts with the gutter pattern.
    2. Extracted line numbers are **strictly monotonically increasing** —
       this is the hallmark of ``read_file`` output and is almost impossible
       to occur in real file content.
    3. The first extracted line number must be >= 1.

    Returns the cleaned text, or ``None`` if any check fails.
    """
    lines = text.split("\n")
    has_prefix = False
    cleaned: list[str] = []
    prev_num = -1
    for line in lines:
        if line == "":
            cleaned.append(line)
            continue
        m = _LINE_NUMBER_PREFIX_RE.match(line)
        if not m:
            return None
        num = _extract_line_number(m)
        if num < 1 or (prev_num >= 0 and num <= prev_num):
            return None
        prev_num = num
        has_prefix = True
        cleaned.append(line[m.end():])
    return "\n".join(cleaned) if has_prefix else None

_FILE_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


def _get_file_lock(full_path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = (id(loop), str(full_path.resolve()))
    lock = _FILE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _FILE_LOCKS[key] = lock
    return lock


def _normalize_newlines(value: str) -> str:
    if not value:
        return value
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _ensure_trailing_newline(content: str) -> str:
    if content and not content.endswith("\n"):
        return content + "\n"
    return content


def _collect_match_positions(haystack: str, needle: str) -> List[int]:
    positions: List[int] = []
    start = 0
    n = len(needle)
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + n
    return positions


def _offset_to_line_number(content: str, offset: int) -> int:
    if offset <= 0:
        return 1
    return content[:offset].count("\n") + 1


def _suggest_fuzzy_block(
    old_block: str, content: str, threshold: float = 0.5
) -> Optional[Tuple[int, str]]:
    normalized_old = _normalize_newlines(old_block)
    normalized_content = _normalize_newlines(content)

    old_lines = normalized_old.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines = old_lines[:-1]
    if not old_lines:
        return None

    content_lines = normalized_content.split("\n")
    n = len(old_lines)
    if n == 0 or n > len(content_lines):
        return None

    target = "\n".join(old_lines)
    best_ratio = 0.0
    best_start = -1
    best_snippet_lines: List[str] = []

    if n == 1:
        max_n = min(_FUZZY_SUGGEST_MAX_LINES, len(content_lines))
        for window_size in range(1, max_n + 1):
            for i in range(len(content_lines) - window_size + 1):
                candidate_lines = content_lines[i : i + window_size]
                candidate = "\n".join(candidate_lines)
                ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_snippet_lines = candidate_lines
    else:
        for i in range(len(content_lines) - n + 1):
            candidate_lines = content_lines[i : i + n]
            candidate = "\n".join(candidate_lines)
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i
                best_snippet_lines = candidate_lines

    if best_ratio >= threshold and best_start >= 0:
        return best_start + 1, "\n".join(best_snippet_lines)
    return None


def _register_edit_tools(mcp):
    """Register file edit tools to FastMCP instance (internal)."""

    @mcp.tool
    async def patch_file(
        file_path: Annotated[
            str,
            Field(
                description=(
                    "File path relative to the session workspace root, or absolute "
                    "starting with '/'."
                ),
            ),
        ],
        old_str: Annotated[
            str,
            Field(
                description=(
                    "Exact substring to find and replace: can be part of a line, a "
                    "full line, or multiple lines. Must match the file "
                    "character-for-character (including spaces and indentation). "
                    "Include enough surrounding context so it appears only once."
                ),
            ),
        ],
        new_str: Annotated[
            str,
            Field(
                description=(
                    "Replacement text (same format as old_str: any contiguous string)."
                ),
            ),
        ],
        require_unique: Annotated[
            bool,
            Field(
                description=(
                    "If True, fail when the pattern matches zero or multiple "
                    "locations; otherwise replace all matches."
                ),
            ),
        ] = True,
    ) -> dict:
        """
        Patch a file by replacing one exact substring with another (find-and-replace).

        Matching is exact after normalising line endings to ``\\n``. When no match
        is found, a fuzzy suggestion may be included in the error. When multiple
        matches exist and ``require_unique`` is True, match line numbers are reported.
        On success, returns a unified diff as ``data``.
        """
        try:
            full_path = resolve_workspace_path(file_path)

            if not full_path.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"File does not exist: {file_path}",
                    )
                )
            if not full_path.is_file():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path is not a file: {file_path}",
                    )
                )

            try:
                async with aiofiles.open(
                    full_path, "r", encoding=_ENCODING_UTF8, errors="ignore"
                ) as f:
                    original_content = await f.read()
            except OSError as exc:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to read file: {exc}",
                    )
                ) from exc

            normalized_content = _normalize_newlines(original_content)
            if normalized_content and not normalized_content.endswith("\n"):
                normalized_content = normalized_content + "\n"

            normalized_old = _normalize_newlines(old_str)
            normalized_new = _normalize_newlines(new_str)

            if not normalized_old:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="old_str must not be empty.",
                    )
                )
            if normalized_old == normalized_new:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="old_str and new_str are identical. No changes needed.",
                    )
                )

            match_positions = _collect_match_positions(normalized_content, normalized_old)

            # Fallback: if no match, check whether old_str carries read_file line-number prefixes
            stripped_old = None
            if not match_positions:
                stripped_old = _strip_line_number_prefixes(normalized_old)
                if stripped_old and stripped_old != normalized_old:
                    match_positions = _collect_match_positions(normalized_content, stripped_old)
                    if match_positions:
                        normalized_old = stripped_old
                        stripped_new = _strip_line_number_prefixes(normalized_new)
                        if stripped_new is not None:
                            normalized_new = stripped_new

            if not match_positions:
                suggestion = _suggest_fuzzy_block(normalized_old, normalized_content)
                msg = (
                    "Pattern not found in file. Ensure old_str matches the current file "
                    "content exactly (including indentation and surrounding context)."
                )
                if stripped_old is not None:
                    msg += " (Line-number prefixes from read_file were detected and stripped, but still no match.)"
                if suggestion is not None:
                    start_line, snippet = suggestion
                    msg += f" Possible nearby block around line {start_line}:\n{snippet}"
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=msg,
                    )
                )

            if require_unique and len(match_positions) != 1:
                line_numbers = [
                    _offset_to_line_number(normalized_content, pos) for pos in match_positions
                ]
                preview = line_numbers[:10]
                msg = f"Pattern matched {len(match_positions)} times at line(s) {preview}"
                if len(line_numbers) > 10:
                    msg += f" (and {len(line_numbers) - 10} more)"
                msg += (
                    ". Provide more specific context in old_str to uniquely locate the target."
                )
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=msg,
                    )
                )

            if require_unique:
                pos = match_positions[0]
                before = normalized_content[:pos]
                after = normalized_content[pos + len(normalized_old) :]
                new_content = before + normalized_new + after
            else:
                new_content = normalized_content.replace(normalized_old, normalized_new)

            new_content = _ensure_trailing_newline(new_content)

            original_for_diff = _ensure_trailing_newline(normalized_content)
            diff_iter = difflib.unified_diff(
                original_for_diff.splitlines(keepends=False),
                new_content.splitlines(keepends=False),
                fromfile=file_path,
                tofile=file_path,
                lineterm="",
            )
            diff_text = "\n".join(diff_iter)

            try:
                lock = _get_file_lock(full_path)
                async with lock:
                    async with aiofiles.open(
                        full_path, "w", encoding=_ENCODING_UTF8
                    ) as f:
                        await f.write(new_content)
                        await f.flush()
            except OSError as exc:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to write file: {exc}",
                    )
                ) from exc

            return {
                "success": True,
                "data": diff_text,
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"patch_file failed: {str(e)}",
                )
            ) from e

    @mcp.tool
    async def file_edit(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        old_lines: Annotated[str, Field(description="Lines to replace/delete as a string. Must match file exactly (whitespace, indentation).")],
        new_lines: Annotated[str, Field(description="New lines to insert/replace as a string.")],
        pre_context: Annotated[str, Field(description="Context lines immediately before old_lines. Must be adjacent and non-overlapping with old_lines. If not provided, no pre context to check.")] = None,
        post_context: Annotated[str, Field(description="Context lines immediately after old_lines. Must be adjacent and non-overlapping with old_lines. If not provided, no post context to check.")] = None,
        require_unique: Annotated[bool, Field(description="If True, fail when the pattern matches zero or multiple locations, otherwise replace all matches.")] = True,
    ) -> dict:
        """
        Edit file by replacing old_lines with new_lines.
        
        Matches contiguous pattern: pre_context (if provided) + old_lines + post_context (if provided). 
        Replace old_lines in the matched pattern with new_lines.
        If not found, file may have changed - read file first.
        """
        try:
            # Validate parameters
            if not pre_context and not post_context and not old_lines:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="file_edit requires at least old_lines, or optionally pre_context/post_context to narrow search."
                    )
                )

            # Resolve client path to workspace
            full_path = await resolve_workspace_path(file_path)

            if not full_path.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"File does not exist: {file_path}"
                    )
                )
            if not full_path.is_file():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path is not a file: {file_path}"
                    )
                )

            # Read original content
            try:
                with full_path.open("r", encoding="utf-8", errors="ignore") as f:
                    original_content = f.read()
            except Exception as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to read file: {e}"
                    )
                )

            # Normalize line endings to '\n'
            if "\r\n" in original_content:
                original_content = original_content.replace("\r\n", "\n")
            elif "\r" in original_content:
                original_content = original_content.replace("\r", "\n")

            had_trailing_newline = original_content.endswith("\n")
            if not had_trailing_newline and original_content:
                original_content = original_content + "\n"

            # Split into lines
            original_lines = original_content.split("\n")

            # Convert string inputs to line lists
            def string_to_lines(value: Optional[str]) -> List[str]:
                if value is None:
                    return []
                return value.replace("\r\n", "\n").replace("\r", "\n").split("\n")

            def strip_trailing_empty_lines(lines: List[str]) -> List[str]:
                """Remove trailing '' from split('\\n') — a string ending with '\\n' always yields an extra empty last segment, which usually should NOT be part of the match (file has '---' or next line, not a blank line)."""
                out = list(lines)
                while out and out[-1] == "":
                    out.pop()
                return out

            # Pattern segments: strip trailing empty lines so e.g. old_lines ending with "...\\n" still matches file where the next line is '---' not a blank line
            s_pre_context = strip_trailing_empty_lines(string_to_lines(pre_context))
            s_post_context = strip_trailing_empty_lines(string_to_lines(post_context))
            s_old_lines = strip_trailing_empty_lines(string_to_lines(old_lines))
            s_new_lines = string_to_lines(new_lines)

            if s_old_lines == s_new_lines:
                return {
                    "success": True,
                    "data": "⚠️ old_lines and new_lines are identical — no changes applied.",
                }

            # Helper: find all indices where a sequence matches
            def find_sequence_indices(haystack: List[str], needle: List[str]) -> List[int]:
                if not needle:
                    return []
                positions = []
                max_start = len(haystack) - len(needle)
                for i in range(max_start + 1):
                    if haystack[i : i + len(needle)] == needle:
                        positions.append(i)
                return positions

            # Build the complete pattern to match (adjacent mode)
            # Pattern: pre_context + old_lines + post_context (all adjacent, no gaps)
            pattern_parts = []
            if s_pre_context:
                pattern_parts.append(("pre_context", s_pre_context))
            if s_old_lines:
                pattern_parts.append(("old_lines", s_old_lines))
            if s_post_context:
                pattern_parts.append(("post_context", s_post_context))

            if not pattern_parts:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message="At least one of pre_context, old_lines, or post_context must be provided."
                    )
                )

            # Build the complete pattern sequence
            complete_pattern = []
            for _, lines in pattern_parts:
                complete_pattern.extend(lines)

            # Find all matches of the complete pattern
            match_positions = find_sequence_indices(original_lines, complete_pattern)

            # Fallback: strip read_file line-number prefixes and retry
            had_line_number_prefix = False
            if not match_positions:
                def _try_strip_lines(lines: List[str]) -> Optional[List[str]]:
                    """Strip line-number prefixes from a list of lines.

                    Returns cleaned lines only when ALL non-empty lines carry a
                    ``read_file``-style gutter AND the extracted line numbers are
                    strictly monotonically increasing (≥ 1).  Returns ``None``
                    otherwise so we never accidentally strip real file content.
                    """
                    cleaned = []
                    found_prefix = False
                    prev_num = -1
                    for line in lines:
                        if line == "":
                            cleaned.append(line)
                            continue
                        m = _LINE_NUMBER_PREFIX_RE.match(line)
                        if not m:
                            return None
                        num = _extract_line_number(m)
                        if num < 1 or (prev_num >= 0 and num <= prev_num):
                            return None
                        prev_num = num
                        found_prefix = True
                        cleaned.append(line[m.end():])
                    return cleaned if found_prefix else None

                stripped_pre = _try_strip_lines(s_pre_context) if s_pre_context else None
                stripped_old = _try_strip_lines(s_old_lines) if s_old_lines else None
                stripped_post = _try_strip_lines(s_post_context) if s_post_context else None

                if stripped_pre is not None or stripped_old is not None or stripped_post is not None:
                    had_line_number_prefix = True
                    if stripped_pre is not None:
                        s_pre_context = stripped_pre
                    if stripped_old is not None:
                        s_old_lines = stripped_old
                    if stripped_post is not None:
                        s_post_context = stripped_post

                    stripped_new = _try_strip_lines(s_new_lines) if s_new_lines else None
                    if stripped_new is not None:
                        s_new_lines = stripped_new

                    complete_pattern = []
                    pattern_parts = []
                    if s_pre_context:
                        pattern_parts.append(("pre_context", s_pre_context))
                    if s_old_lines:
                        pattern_parts.append(("old_lines", s_old_lines))
                    if s_post_context:
                        pattern_parts.append(("post_context", s_post_context))
                    for _, lines in pattern_parts:
                        complete_pattern.extend(lines)
                    match_positions = find_sequence_indices(original_lines, complete_pattern)

            if not match_positions:
                # Build error message describing what pattern was searched for
                pattern_desc_parts = []
                if s_pre_context:
                    pattern_desc_parts.append("pre_context")
                if s_old_lines:
                    pattern_desc_parts.append("old_lines")
                if s_post_context:
                    pattern_desc_parts.append("post_context")
                pattern_desc = " + ".join(pattern_desc_parts)
                
                error_msg = f"Pattern not found: {pattern_desc}. The pattern must appear as a continuous sequence in the file."
                if had_line_number_prefix:
                    error_msg += " (Line-number prefixes from read_file were detected and stripped, but still no match.)"
                if s_pre_context:
                    pre_found = find_sequence_indices(original_lines, s_pre_context)
                    if pre_found:
                        error_msg += f" pre_context found at line(s) {[p + 1 for p in pre_found]}."
                    else:
                        error_msg += " pre_context not found in file."
                if s_post_context:
                    post_found = find_sequence_indices(original_lines, s_post_context)
                    if post_found:
                        error_msg += f" post_context found at line(s) {[p + 1 for p in post_found]}."
                    else:
                        error_msg += " post_context not found in file."

                # Fuzzy suggestion: show the closest matching block so the
                # agent can see where its old_lines diverged from the file.
                fuzzy_target = "\n".join(s_old_lines) if s_old_lines else "\n".join(complete_pattern)
                if fuzzy_target:
                    suggestion = _suggest_fuzzy_block(fuzzy_target, original_content)
                    if suggestion is not None:
                        sug_line, sug_snippet = suggestion
                        # Truncate very long snippets to keep the error readable
                        if len(sug_snippet) > 600:
                            sug_snippet = sug_snippet[:600] + "\n... (truncated)"
                        error_msg += f"\n\nDid you mean this block near line {sug_line}?\n{sug_snippet}"
                    else:
                        error_msg += "\n\nNo similar block found in the file. Use read_file to get the exact content before editing."

                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=error_msg
                    )
                )

            if require_unique and len(match_positions) != 1:
                match_line_numbers = [pos + 1 for pos in match_positions]
                error_msg = f"Pattern matched {len(match_positions)} times at line(s) {match_line_numbers[:10]}"
                if len(match_positions) > 10:
                    error_msg += f" (and {len(match_positions) - 10} more)"
                error_msg += ". Provide more surrounding context in old_lines, or use pre_context/post_context to uniquely locate the target."

                # Show a short surrounding-context preview for each match so
                # the agent can figure out which location it intended.
                for idx, pos in enumerate(match_positions[:5]):
                    ctx_start = max(0, pos - 1)
                    ctx_end = min(len(original_lines), pos + len(complete_pattern) + 1)
                    snippet_lines = original_lines[ctx_start:ctx_end]
                    snippet = "\n".join(snippet_lines)
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    error_msg += f"\n\n--- Match {idx + 1} at line {pos + 1} ---\n{snippet}"

                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=error_msg
                    )
                )

            # Calculate target position
            target_start = match_positions[0]
            
            # Determine what to replace:
            # - If old_lines is provided, replace only old_lines (within the matched pattern)
            # - If old_lines is not provided, replace the entire matched pattern
            if s_old_lines:
                # Find where old_lines starts within the complete pattern
                pattern_offset = 0
                if s_pre_context:
                    pattern_offset = len(s_pre_context)
                # old_lines starts at pattern_offset within the complete pattern
                target_start = target_start + pattern_offset
                target_end = target_start + len(s_old_lines)
            else:
                # Replace the entire matched pattern
                target_end = target_start + len(complete_pattern)

            # Build new content
            new_content_lines = original_lines[:target_start] + s_new_lines + original_lines[target_end:]

            # Join back into content
            new_content = "\n".join(new_content_lines)
            if not new_content.endswith("\n") and new_content:
                new_content = new_content + "\n"

            # Write back to file
            try:
                with full_path.open("w", encoding="utf-8") as f:
                    f.write(new_content)
            except PermissionError:
                await _try_repair_docker_owned_path(full_path)
                try:
                    with full_path.open("w", encoding="utf-8") as f:
                        f.write(new_content)
                except Exception as e:
                    raise McpError(
                        ErrorData(
                            code=INTERNAL_ERROR,
                            message=f"Failed to write file: {e}"
                        )
                    )
            except Exception as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to write file: {e}"
                    )
                )

            affected_old_lines = len(s_old_lines)
            affected_new_lines = len(s_new_lines)
            start_line = target_start + 1  # Convert to 1-based
            end_line = target_end

            # Generate concise message
            if affected_old_lines == 0 and affected_new_lines > 0:
                message = f"Successfully inserted {affected_new_lines} line(s) at line {start_line} in {file_path}"
            elif affected_old_lines > 0 and affected_new_lines == 0:
                message = f"Successfully deleted {affected_old_lines} line(s) (lines {start_line}-{end_line}) in {file_path}"
            else:
                message = f"Successfully replaced {affected_old_lines} line(s) with {affected_new_lines} line(s) (lines {start_line}-{end_line}) in {file_path}"

            return {
                "success": True,
                "data": message,
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"file_edit failed: {str(e)}"
                )
            )

    @mcp.tool
    async def replace_file_block(
        file_path: Annotated[str, Field(description="File path relative to the session workspace root, or absolute starting with '/'")],
        start_line: Annotated[int, Field(description="Start line number (inclusive, 1-based)", ge=1)],
        end_line: Annotated[int, Field(description="End line number (inclusive, 1-based)", ge=1)],
        new_block_content: Annotated[str, Field(description="New content to insert/replace between start_line and end_line.")],
    ) -> dict:
        """
        Replace file content in range [start_line, end_line] with new_block_content.
        Both line numbers are 1-based and inclusive.
        """
        try:
            # Validate parameters
            if start_line < 1 or end_line < 1:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Line numbers must be >= 1, got start_line={start_line}, end_line={end_line}"
                    )
                )

            # Resolve client path to workspace
            full_path = await resolve_workspace_path(file_path)

            if not full_path.exists():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"File does not exist: {file_path}"
                    )
                )
            if not full_path.is_file():
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"Path is not a file: {file_path}"
                    )
                )

            # Read file lines
            try:
                with full_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to read file: {e}"
                    )
                )

            total_lines = len(lines)

            # Handle empty file
            if total_lines == 0:
                # Write new content directly
                full_path.write_text(new_block_content, encoding="utf-8")
                return {
                    "success": True,
                    "data": f"Content written to empty file: {file_path}",
                }

            # Validate line numbers
            if end_line > total_lines:
                end_line = total_lines
            
            if start_line > end_line:
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message=f"start_line ({start_line}) must be <= end_line ({end_line})"
                    )
                )

            # Prepare new content lines (ensure each logical line ends with \n so we never
            # glue the last inserted line to the first preserved line below the range).
            if new_block_content == "":
                new_block_lines = []
            else:
                new_block_lines = new_block_content.splitlines(keepends=True)
                if new_block_lines and not new_block_lines[-1].endswith(("\n", "\r")):
                    new_block_lines[-1] += "\n"

            # Perform edit operation (convert to 0-based indexing)
            start_idx = start_line - 1
            end_idx = end_line - 1  # inclusive end
            
            # Replace lines [start_line, end_line] (inclusive) with new_block_content
            new_lines = lines[:start_idx] + new_block_lines + lines[end_idx + 1:]

            # Write modified content back
            with full_path.open("w", encoding="utf-8") as f:
                f.writelines(new_lines)

            affected_range = f"line {start_line}" if start_line == end_line else f"lines {start_line}-{end_line}"
            new_line_count = len(new_block_lines)

            return {
                "success": True,
                "data": f"Successfully replaced {affected_range} in {file_path} ({new_line_count} new lines)",
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"replace_file_block failed: {str(e)}"
                )
            )
