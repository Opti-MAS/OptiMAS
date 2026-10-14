"""
File lookup operations tools.

Provides:
- grep_file_content: Search for content in text files matching a pattern, with optional replacement.
  Read-only search uses **ripgrep (rg)** when available (JSON stream, default exclude globs,
  30s soft timeout, broad-pattern guard); otherwise falls back to the Python scanner.
"""
import asyncio
import fnmatch
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated, DefaultDict, Dict, List, Optional, Tuple

from mcp import McpError
from mcp.types import ErrorData
from pydantic import Field

from src.tools.common_utils.path_utils import (
    get_session_workspace,
    resolve_workspace_path,
    get_client_path,
)
from src.utils.error_codes import INVALID_PARAMS, INTERNAL_ERROR

_GREP_SOFT_TIMEOUT_SECONDS = 30.0
# After terminate/kill, do not block longer than this reaping `rg` (avoids hung MCP calls).
_RG_PROC_REAP_TIMEOUT_SECONDS = 30.0

_REGEX_METACHAR_RE = re.compile(
    r"(?:"
    r"\\[tdwsnDWSbB]"       # common regex escapes: \t \d \w \s \n etc.
    r"|(?<!\\)\[.+?(?<!\\)\]"   # character classes: [A-Z], [0-9], [^abc]
    r"|(?<!\\)[+?]"         # quantifiers: + ?
    r"|(?<!\\)\.\*"         # wildcard: .*
    r"|(?<!\\)\("           # groups: (
    r"|(?<!\\)\|"           # alternation: |
    r"|\{\d+(?:,\d*)?\}"   # repetition: {3}, {1,5}
    r")"
)


def _looks_like_regex(pattern: str) -> bool:
    """Heuristic: return True if the pattern contains common regex metacharacters."""
    return bool(_REGEX_METACHAR_RE.search(pattern))


_GREP_DEFAULT_EXCLUDE_GLOBS: Tuple[str, ...] = (
    "!**/_internal/**",
    "!pnpm-lock.yaml",
    "!package-lock.json",
    "!yarn.lock",
    "!.eslintrc",
    "!.eslintrc.*",
    "!eslint.config.*",
    "!.prettierrc",
    "!.prettierrc.*",
    "!prettier.config.*",
    "!tsconfig*.json",
    "!vite.config.*",
)

_rg_available: Optional[bool] = None


def _is_rg_available() -> bool:
    global _rg_available
    if _rg_available is None:
        _rg_available = shutil.which("rg") is not None
    return bool(_rg_available)


def _check_grep_pattern_too_broad(
    pattern: str,
    path: Optional[str],
    include_glob: Optional[str],
) -> None:
    """Reject trivially dangerous single-character workspace-wide searches."""
    if len(pattern) != 1 or pattern not in {"<", ">", "="}:
        return
    path_is_root = path is None or (isinstance(path, str) and path.strip() in ("", "."))
    inc = (include_glob or "").strip()
    include_is_global = inc in ("", "*", "**/*")
    if path_is_root and include_is_global:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=(
                    "grep pattern is too broad and may scan the entire workspace "
                    "(e.g. searching for a single character like '<' without "
                    "restricting 'path' or 'include_glob'). Please narrow your "
                    "search by setting 'path' to a specific subdirectory "
                    "(such as 'src/pages') and using a more specific pattern."
                ),
            )
        )


def _parse_rg_path(data: dict, workspace: Path) -> Tuple[Optional[Path], int, str]:
    path_raw = data.get("path")
    if isinstance(path_raw, str):
        path_text = path_raw.strip()
    elif path_raw and isinstance(path_raw, dict):
        path_text = (path_raw.get("text") or "").strip()
    else:
        path_text = ""
    line_number = int(data.get("line_number") or 0)
    lines_data = data.get("lines") or {}
    line_content = (
        lines_data.get("text") if isinstance(lines_data, dict) else ""
    ) or ""
    if isinstance(line_content, str):
        line_content = line_content.rstrip("\n\r")
    else:
        line_content = ""
    if not path_text or line_number < 1:
        return None, 0, ""
    path_text = path_text.replace("\\", "/")
    abs_path = (workspace / path_text).resolve()
    return abs_path, line_number, line_content


def _merge_overlapping_contexts(
    matches_by_file: Dict[Path, List[int]],
    lines_before: int,
    lines_after: int,
) -> List[Tuple[Path, int, int, int]]:
    result: List[Tuple[Path, int, int, int]] = []
    for path_obj, line_nos in matches_by_file.items():
        if not line_nos:
            continue
        line_nos_sorted = sorted(set(line_nos))
        intervals: List[Tuple[int, int]] = []
        for ln in line_nos_sorted:
            intervals.append((max(1, ln - lines_before), ln + lines_after))
        merged: List[Tuple[int, int]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            rep = next(ln for ln in line_nos_sorted if s <= ln <= e)
            result.append((path_obj, s, e, rep))
    return result


def _read_file_slice_sync(path: Path, start_line: int, end_line: int) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (UnicodeDecodeError, PermissionError, OSError):
        return out
    for i in range(max(0, start_line - 1), min(len(lines), end_line)):
        out.append((i + 1, lines[i].rstrip("\n\r")))
    return out


async def _run_rg_search(
    workspace: Path,
    pattern: str,
    path_arg: Optional[str],
    max_results: int,
    is_regex: bool,
    case_sensitive: bool,
    include_glob: Optional[str],
    exclude_glob: Optional[str],
) -> List[Tuple[Path, int, str]]:
    workspace_resolved = workspace.resolve()
    if not workspace_resolved.is_dir():
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"Workspace is not a directory: {workspace_resolved}",
            )
        )

    if path_arg is None or (isinstance(path_arg, str) and path_arg.strip() == ""):
        search_target = "."
    else:
        target = await resolve_workspace_path(path_arg.strip())
        if not target.exists():
            raise McpError(
                ErrorData(
                    code=INVALID_PARAMS,
                    message=f"Path does not exist: {path_arg}",
                )
            )
        try:
            rel = target.resolve().relative_to(workspace_resolved)
            search_target = rel.as_posix()
        except ValueError:
            search_target = "."

    argv: List[str] = [
        "rg",
        "--json",
        "-n",
        "--no-heading",
        "--max-filesize",
        "10M",
    ]
    for g in _GREP_DEFAULT_EXCLUDE_GLOBS:
        argv.extend(["--glob", g])
    if include_glob and include_glob.strip():
        argv.extend(["--glob", include_glob.strip()])
    if exclude_glob and exclude_glob.strip():
        argv.extend(["--glob", f"!{exclude_glob.strip()}"])
    if not case_sensitive:
        argv.append("-i")
    if not is_regex:
        argv.append("-F")
    # Use POSIX ``--`` to terminate flag parsing so patterns starting with ``-``
    # (e.g. ``--recursive``, ``-i``, ``-F``) are interpreted as the search
    # string rather than misparsed as ripgrep flags.
    argv.append("--")
    argv.append(pattern)
    argv.append(search_target)

    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(workspace_resolved),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    matches: List[Tuple[Path, int, str]] = []
    try:
        if proc.stdout is None:
            return matches
        while len(matches) < max_results:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line_str = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
            if not line_str:
                continue
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "match":
                continue
            data = obj.get("data") or {}
            abs_path, line_number, line_content = _parse_rg_path(data, workspace_resolved)
            if abs_path is None:
                continue
            try:
                abs_path.relative_to(workspace_resolved)
            except ValueError:
                continue
            matches.append((abs_path, line_number, line_content))
    finally:
        # Always reap `rg`. If the caller's ``wait_for(..., _GREP_SOFT_TIMEOUT_SECONDS)`` cancels
        # this coroutine, we must stop the subprocess: otherwise ``await proc.wait()`` would block
        # until ripgrep finishes scanning the whole workspace (minutes+), and the MCP client then
        # fails with a long tool timeout (~900s) instead of the intended soft timeout error.
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=_RG_PROC_REAP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

    if proc.returncode == 2:
        detail = ""
        if proc.stderr:
            detail = (await proc.stderr.read()).decode("utf-8", errors="replace").strip()
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message=f"ripgrep failed: {detail or 'unknown error'}",
            )
        )
    return matches


def _filter_paths_by_globs(
    paths: List[Path],
    workspace: Path,
    include_glob: Optional[str],
    exclude_glob: Optional[str],
) -> List[Path]:
    """Best-effort fnmatch filter for the Python scanner (rg uses native globs)."""
    inc = (include_glob or "").strip()
    exc = (exclude_glob or "").strip()
    if not inc and not exc:
        return paths
    w = workspace.resolve()
    out: List[Path] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(w).as_posix()
        except ValueError:
            continue
        if exc and (fnmatch.fnmatch(rel, exc) or fnmatch.fnmatch(rel.split("/")[-1], exc)):
            continue
        if inc and not (fnmatch.fnmatch(rel, inc) or fnmatch.fnmatch(rel.split("/")[-1], inc)):
            continue
        out.append(p)
    return out


async def _grep_file_content_via_ripgrep(
    workspace: Path,
    pattern: str,
    path: Optional[str],
    lines_before: int,
    lines_after: int,
    max_results: int,
    interpret_as_regex: bool,
    case_sensitive: bool,
    include_glob: Optional[str],
    exclude_glob: Optional[str],
) -> List[dict]:
    """Build the same structured result list as the Python scanner."""
    rg_case_sensitive = case_sensitive or any(c.isupper() for c in pattern)

    try:
        matches = await asyncio.wait_for(
            _run_rg_search(
                workspace,
                pattern,
                path,
                max_results,
                interpret_as_regex,
                rg_case_sensitive,
                include_glob,
                exclude_glob,
            ),
            timeout=_GREP_SOFT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=(
                    f"grep_file_content timed out after approximately "
                    f"{int(_GREP_SOFT_TIMEOUT_SECONDS)} seconds. Narrow 'path', "
                    "'include_glob', or use a more specific pattern."
                ),
            )
        ) from exc

    matches_by_file: DefaultDict[Path, List[int]] = defaultdict(list)
    for abs_path, line_no, _ in matches:
        matches_by_file[abs_path].append(line_no)
    merged = _merge_overlapping_contexts(
        dict(matches_by_file), lines_before, lines_after
    )

    structured_results: List[dict] = []
    for path_abs, start_line, end_line, rep_line in merged:
        match_lines = {
            ln
            for (p, ln, _) in matches
            if p == path_abs and start_line <= ln <= end_line
        }
        slice_lines = await asyncio.to_thread(
            _read_file_slice_sync, path_abs, start_line, end_line
        )
        try:
            rel_key = "/" + await get_client_path(path_abs)
        except McpError:
            rel_key = str(path_abs)

        context_parts = []
        for ln, content in slice_lines:
            prefix = ">>> " if ln in match_lines else "    "
            context_parts.append(f"{prefix}{ln}|{content}")
        context_text = "\n".join(context_parts)
        file_path_with_line = f"{rel_key}:{rep_line}"
        structured_results.append(
            {"file_path": file_path_with_line, "context": context_text}
        )

    if len(matches) >= max_results:
        cap = min(2 * max_results, 100)
        try:
            count_matches = await asyncio.wait_for(
                _run_rg_search(
                    workspace,
                    pattern,
                    path,
                    cap,
                    interpret_as_regex,
                    rg_case_sensitive,
                    include_glob,
                    exclude_glob,
                ),
                timeout=_GREP_SOFT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            count_matches = matches

        if len(count_matches) < cap:
            total = len(count_matches)
            files_count = len({p for p, _, _ in count_matches})
            note = (
                f"[summary] {total} match(es) in {files_count} file(s); "
                f"showing first {max_results} only. Narrow path or include_glob."
            )
        else:
            note = (
                f"[summary] More than {cap} matches; showing first {max_results} only. "
                "Use a more specific pattern or path."
            )
        if structured_results:
            structured_results[0]["context"] = note + "\n" + structured_results[0]["context"]
        else:
            structured_results.append({"file_path": "/_grep_note", "context": note})

    return structured_results


def _register_search_tools(mcp):
    """Register file lookup tools to FastMCP instance (internal)."""

    @mcp.tool
    async def grep_file_content(
        pattern: Annotated[str, Field(description=(
            "Search pattern. By default this is a literal text search. "
            "For regex patterns (e.g. '\\d+', '[A-Z]', '.*error.*'), set interpret_as_regex=true. "
            "Note: patterns containing regex metacharacters like [], \\t, \\d, +, ?, .* are "
            "auto-promoted to regex mode even without interpret_as_regex. "
            "Prefer simple keywords or short phrases for best results."
        ))],
        replacement: Annotated[str, Field(description="Optional. If provided, perform search-and-replace and return results in grep-like format.")] = "",
        path: Annotated[str, Field(description=(
            "File or directory path (relative to workspace). Strongly recommended to narrow search "
            "scope. If omitted, searches all text files in the workspace."
        ))] = None,
        lines_before: Annotated[int, Field(description="Context lines before each match.", ge=0, le=20)] = 3,
        lines_after: Annotated[int, Field(description="Context lines after each match.", ge=0, le=20)] = 3,
        max_results: Annotated[int, Field(description="Maximum number of matches to return.", ge=1, le=100)] = 15,
        interpret_as_regex: Annotated[bool, Field(description=(
            "Set to true to interpret pattern as a regex. "
            "Auto-enabled when pattern contains obvious regex syntax ([], \\t, \\d, +, etc.)."
        ))] = False,
        case_sensitive: Annotated[bool, Field(description="Whether the search is case sensitive. Default false (case-insensitive).")] = False,
        include_glob: Annotated[
            str,
            Field(
                description="Optional glob to include files (e.g. 'src/**/*.py'). "
                "Applied natively when using ripgrep; best-effort fnmatch on the Python path.",
                default=None,
            ),
        ] = None,
        exclude_glob: Annotated[
            str,
            Field(
                description="Optional glob to exclude files (e.g. '*.min.js').",
                default=None,
            ),
        ] = None,
    ) -> dict:
        """Search for content in text files matching the pattern.

        Returns matching lines with surrounding context. Use simple keywords or
        short phrases for literal search; use interpret_as_regex=true (or regex
        syntax which is auto-detected) for advanced pattern matching.
        If replacement is provided, replace matches and show results in grep format.
        """
        try:
            # Auto-promote to regex mode when the pattern clearly contains
            # regex metacharacters but the caller forgot interpret_as_regex.
            if not interpret_as_regex and _looks_like_regex(pattern):
                interpret_as_regex = True

            # Compile regex if needed
            if interpret_as_regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    regex = re.compile(pattern, flags)
                    use_regex = True
                except re.error as e:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=f"Invalid regex pattern: {str(e)}",
                        )
                    )
            else:
                regex = None
                use_regex = False

            # Get session workspace
            workspace = await get_session_workspace()
            _check_grep_pattern_too_broad(pattern, path, include_glob)

            # If replacement is provided, perform replace operation
            if replacement:
                # Resolve path if provided
                if path:
                    target_path = await resolve_workspace_path(path)
                    if target_path.is_file():
                        file_paths = [target_path]
                    elif target_path.is_dir():
                        # Get all files in directory recursively
                        file_paths = [
                            p for p in target_path.rglob("*") if p.is_file()
                        ]
                    else:
                        raise McpError(
                            ErrorData(
                                code=INVALID_PARAMS,
                                message=f"Path does not exist: {path}",
                            )
                        )
                else:
                    # Search all files in workspace
                    file_paths = [p for p in workspace.rglob("*") if p.is_file()]

                # Filter text files (simplified: check if readable as text)
                text_file_paths = []
                for p in file_paths:
                    try:
                        # Skip large files (>10MB)
                        if p.stat().st_size > 10 * 1024 * 1024:
                            continue
                        # Try to read as text
                        with p.open("r", encoding="utf-8", errors="ignore") as f:
                            f.read(1024)  # Try reading first 1KB
                        text_file_paths.append(p)
                    except (UnicodeDecodeError, PermissionError, OSError):
                        continue

                text_file_paths = _filter_paths_by_globs(
                    text_file_paths, workspace, include_glob, exclude_glob
                )

                # Process each file: search for matches, replace, and collect results
                total_replacements = 0
                files_modified = 0
                results = []

                for file_path in text_file_paths:
                    # Read file content
                    try:
                        with file_path.open(
                            "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            original_content = f.read()
                    except (UnicodeDecodeError, PermissionError, OSError):
                        continue

                    # Normalize line endings
                    if "\r\n" in original_content:
                        original_content = original_content.replace("\r\n", "\n")
                    elif "\r" in original_content:
                        original_content = original_content.replace("\r", "\n")

                    # Count matches and perform replacement
                    if use_regex:
                        match_count = len(regex.findall(original_content))
                        if match_count == 0:
                            continue
                        new_content = regex.sub(replacement, original_content)
                    else:
                        if case_sensitive:
                            match_count = original_content.count(pattern)
                            if match_count == 0:
                                continue
                            new_content = original_content.replace(pattern, replacement)
                        else:
                            # Case-insensitive search and replace
                            pattern_lower = pattern.lower()
                            content_lower = original_content.lower()
                            match_count = content_lower.count(pattern_lower)
                            if match_count == 0:
                                continue
                            # Replace with case preservation
                            new_content = re.sub(
                                re.escape(pattern),
                                replacement,
                                original_content,
                                flags=re.IGNORECASE
                            )

                    # Write back to file
                    try:
                        with file_path.open("w", encoding="utf-8") as f:
                            f.write(new_content)
                    except (PermissionError, OSError):
                        continue

                    total_replacements += match_count
                    files_modified += 1

                    # Show results in grep format: find all lines that were replaced
                    modified_lines = new_content.split("\n")
                    original_lines = original_content.split("\n")

                    # Find lines that were modified
                    for i in range(min(len(original_lines), len(modified_lines))):
                        orig_line = original_lines[i].rstrip("\n\r")
                        mod_line = modified_lines[i].rstrip("\n\r")

                        # Check if this line was modified
                        was_modified = False
                        if use_regex:
                            was_modified = (regex.search(orig_line) is not None) or (
                                replacement in mod_line and orig_line != mod_line
                            )
                        else:
                            if case_sensitive:
                                was_modified = (pattern in orig_line) or (
                                    replacement in mod_line and orig_line != mod_line
                                )
                            else:
                                was_modified = (pattern.lower() in orig_line.lower()) or (
                                    replacement in mod_line and orig_line != mod_line
                                )

                        if was_modified and len(results) < max_results:
                            start_line = max(0, i - lines_before)
                            end_line = min(len(modified_lines), i + lines_after + 1)

                            context_lines = []
                            for j in range(start_line, end_line):
                                ctx_content = (
                                    modified_lines[j].rstrip("\n\r")
                                    if j < len(modified_lines)
                                    else ""
                                )
                                is_match = j == i
                                context_lines.append(
                                    {
                                        "line_number": j + 1,
                                        "content": ctx_content,
                                        "is_match": is_match,
                                    }
                                )

                            # Get relative path for display
                            try:
                                rel_key = "/" + await get_client_path(file_path)
                            except McpError:
                                rel_key = str(file_path)

                            results.append(
                                {
                                    "file": rel_key,
                                    "line_number": i + 1,
                                    "context": context_lines,
                                }
                            )

                # Build structured output
                if total_replacements == 0:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message="No matches found for the given pattern.",
                        )
                    )

                # Build structured results
                structured_results = []
                for result in results:
                    # Build context string with line numbers (match line is included in context)
                    context_parts = []
                    for ctx_line in result["context"]:
                        line_str = f"{ctx_line['line_number']}|{ctx_line['content']}"
                        context_parts.append(line_str)
                    
                    context_text = "\n".join(context_parts)
                    
                    # Format file_path as "file_path:line_number" (grep-style)
                    # The line_number in file_path indicates which line in context is the match
                    file_path_with_line = f"{result['file']}:{result['line_number']}"

                    structured_results.append({
                        "file_path": file_path_with_line,
                        "context": context_text,
                    })

                return {
                    "success": True,
                    "data": structured_results,
                }

            # Normal grep search (no replacement)
            if _is_rg_available():
                structured = await _grep_file_content_via_ripgrep(
                    workspace,
                    pattern,
                    path,
                    lines_before,
                    lines_after,
                    max_results,
                    interpret_as_regex,
                    case_sensitive,
                    include_glob,
                    exclude_glob,
                )
                return {"success": True, "data": structured}

            # Resolve path if provided
            if path:
                target_path = await resolve_workspace_path(path)
                if target_path.is_file():
                    file_paths = [target_path]
                elif target_path.is_dir():
                    file_paths = [p for p in target_path.rglob("*") if p.is_file()]
                else:
                    raise McpError(
                        ErrorData(
                            code=INVALID_PARAMS,
                            message=f"Path does not exist: {path}",
                        )
                    )
            else:
                # Search all files in workspace
                file_paths = [p for p in workspace.rglob("*") if p.is_file()]

            # Filter text files
            text_file_paths = []
            for p in file_paths:
                try:
                    if p.stat().st_size > 10 * 1024 * 1024:
                        continue
                    with p.open("r", encoding="utf-8", errors="ignore") as f:
                        f.read(1024)
                    text_file_paths.append(p)
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue

            text_file_paths = _filter_paths_by_globs(
                text_file_paths, workspace, include_glob, exclude_glob
            )

            results = []
            total_matches = 0

            for file_path in text_file_paths:
                if total_matches >= max_results:
                    break

                # Read file content
                try:
                    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue

                # Search for matches in the file
                for i, line in enumerate(lines):
                    if total_matches >= max_results:
                        break

                    line_content = line.rstrip("\n\r")

                    # Match based on mode: regex or literal string
                    if use_regex:
                        is_match = regex.search(line_content) is not None
                    else:
                        if case_sensitive:
                            is_match = pattern in line_content
                        else:
                            is_match = pattern.lower() in line_content.lower()

                    if is_match:
                        total_matches += 1

                        # Calculate context range
                        start_line = max(0, i - lines_before)
                        end_line = min(len(lines), i + lines_after + 1)

                        # Get context lines
                        context_lines = []
                        for j in range(start_line, end_line):
                            line_number = j + 1
                            ctx_content = lines[j].rstrip("\n\r")
                            is_match_line = j == i
                            context_lines.append(
                                {
                                    "line_number": line_number,
                                    "content": ctx_content,
                                    "is_match": is_match_line,
                                }
                            )

                        # Get relative path for display
                        try:
                            rel_key = "/" + await get_client_path(file_path)
                        except McpError:
                            rel_key = str(file_path)

                        results.append(
                            {
                                "file": rel_key,
                                "line_number": i + 1,
                                "match": line_content,
                                "context": context_lines,
                            }
                        )

            # Build structured results
            structured_results = []
            for result in results:
                # Build context string with line numbers (match line is included in context)
                context_parts = []
                for ctx_line in result["context"]:
                    line_str = f"{ctx_line['line_number']}|{ctx_line['content']}"
                    context_parts.append(line_str)

                context_text = "\n".join(context_parts)
                
                # Format file_path as "file_path:line_number" (grep-style)
                # The line_number in file_path indicates which line in context is the match
                file_path_with_line = f"{result['file']}:{result['line_number']}"

                structured_results.append({
                    "file_path": file_path_with_line,
                    "context": context_text,
                })

            return {
                "success": True,
                "data": structured_results,
            }

        except McpError:
            raise
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"grep_file_content failed: {str(e)}",
                )
            )

