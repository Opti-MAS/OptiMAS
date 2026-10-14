"""Internal AI tools: text_qa, text_extract, and image_qa.

Design principles:
  - Safety first: reject binary/unsupported formats, sanitize input, cap output length.
  - Large files: sliding-window QA — each chunk updates a cumulative answer; the last
    LLM turn is returned (no separate synthesis pass). All chunks are processed.
  - text_extract: multi-file structured dimension extraction (JSON) with halo-chunked
    LLM calls and bounded concurrency (MCP read + ``llm.generate``).
  - Auto-summary: if the final answer exceeds a threshold, compress it automatically.
  - URL support (image_qa): download remote images to a workspace-local cache.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .prompt_utils import load_prompt_template

if TYPE_CHECKING:
    from ...mcp.client import MCPClient
    from ...llm.base import BaseLLM

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts" / "internal_ai_tools"
_TEXT_QA_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "text_qa_system_prompt.txt"
_TEXT_EXTRACT_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "text_extract_system_prompt.txt"
_TEXT_EXTRACT_USER_MESSAGE_PATH = _PROMPTS_DIR / "text_extract_user_message.txt"
_IMAGE_QA_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "image_qa_system_prompt.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_CHARS = 16_000
MAX_OUTPUT_CHARS = 8_000
# Sub-windows for very large read_file_for_qa segments (single logical chunk can exceed LLM budget)
_MAX_EXTRACT_CHARS = 14_000
_EXTRACT_OVERLAP = 400

# text_extract: halo chunking + parallelism (aligned with parallel-project defaults)
_DEFAULT_TEXT_EXTRACT_CORE_WINDOW = 50_000
_DEFAULT_TEXT_EXTRACT_HALO = 200
_DEFAULT_TEXT_EXTRACT_MAX_CHUNKS_PER_DOC = 20
_DEFAULT_TEXT_EXTRACT_LLM_CONCURRENCY = 20

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

BLOCKED_TEXT_SUFFIXES = {
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff",
}

# Rich formats supported by MCP read_file via rich_file_reader extraction.
# text_qa uses a dedicated path for these: full-text extraction then QA.
RICH_TEXT_SUFFIXES = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".csv", ".html", ".htm",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sanitize_text(text: str) -> str:
    """Remove null bytes and other characters that may crash LLM APIs."""
    return text.replace("\x00", "")


async def _auto_summary(
    llm: "BaseLLM",
    text: str,
    question: str,
    max_chars: int = MAX_OUTPUT_CHARS,
) -> str:
    """If *text* exceeds *max_chars*, ask the LLM to compress it."""
    if len(text) <= max_chars:
        return text
    try:
        messages = [
            {"role": "system", "content": (
                "Compress the following answer into a concise summary that preserves "
                "all key facts, numbers, names, and conclusions. Stay under "
                f"{max_chars} characters. Do NOT omit important details."
            )},
            {"role": "user", "content": (
                f"Original question: {question}\n\n"
                f"Long answer to compress:\n{text[:min(len(text), max(max_chars * 8, 96_000))]}"
            )},
        ]
        resp = await llm.generate(messages=messages)
        summary = resp.get("text_content", "").strip()
        return summary if summary else text[:max_chars] + "\n...[truncated]"
    except Exception as e:
        logger.warning("auto_summary failed: %s — truncating instead", e)
        return text[:max_chars] + "\n...[truncated]"


async def _read_file_lines(
    mcp_client: "MCPClient",
    file_path: str,
    start_line: int,
    end_line: Optional[int] = None,
) -> Dict[str, Any]:
    """Read a range of lines from a file via MCP ``read_file``."""
    args: Dict[str, Any] = {
        "file_path": file_path,
        "include_line_numbers": False,
        "start_line": start_line,
    }
    if end_line is not None:
        args["end_line"] = end_line
    return await mcp_client.call_tool("read_file", args)


def _extract_content(result: Dict[str, Any]) -> str:
    """Extract text content from an MCP read_file result."""
    if not result.get("success"):
        return ""
    data = result.get("data", "")
    if isinstance(data, dict):
        return _sanitize_text(data.get("content", str(data)))
    text = str(data)
    m = re.search(r"<file_content>(.*?)</file_content>", text, re.DOTALL)
    if m:
        return _sanitize_text(m.group(1))
    return _sanitize_text(text)


def _parse_total_lines_from_read_file_payload(data: Any) -> int:
    """Parse ``Total lines in file: N`` prefix emitted by MCP ``read_file``."""
    m = re.search(r"Total lines in file:\s*(\d+)", str(data or ""), re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def _compute_line_chunks(total_lines: int, lines_per_chunk: int) -> List[tuple]:
    """Return list of (start_line, end_line) 1-based inclusive ranges."""
    chunks = []
    for start in range(1, total_lines + 1, lines_per_chunk):
        end = min(start + lines_per_chunk - 1, total_lines)
        chunks.append((start, end))
    return chunks


async def _text_qa_via_read_file_for_qa_chunks(
    mcp_client: "MCPClient",
    llm: "BaseLLM",
    file_path: str,
    question: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Large / uneven lines (e.g. JSONL with a huge first row): MCP ``read_file_for_qa`` (~96k per chunk).

    Every logical chunk (and each length-split sub-segment) updates one cumulative answer;
    the last model response is the tool result (then optional auto-summary).
    """
    running_answer = ""
    boundaries_1based: Optional[List[int]] = None
    total_chunks = 1
    file_chunks_done = 0

    idx = 1
    while idx <= total_chunks:
        args: Dict[str, Any] = {"file_path": file_path, "chunk_index": idx}
        if boundaries_1based is not None and idx > 1:
            args["start_line"] = int(boundaries_1based[idx - 1])
            args["end_line"] = int(boundaries_1based[idx]) - 1

        res = await mcp_client.call_tool("read_file_for_qa", args)
        if not res.get("success"):
            return {
                "success": False,
                "data": None,
                "error": str(res.get("error") or "read_file_for_qa failed"),
            }

        payload = res.get("data")
        if not isinstance(payload, dict):
            return {
                "success": False,
                "data": None,
                "error": "read_file_for_qa returned unexpected payload (tool missing on server?)",
            }

        if idx == 1:
            b = payload.get("chunk_boundaries")
            if isinstance(b, list) and len(b) >= 2:
                boundaries_1based = [int(x) for x in b]
                total_chunks = len(boundaries_1based) - 1
            else:
                progress = str(payload.get("progress") or "")
                mprog = re.search(r"/(\d+)\s*chunks", progress)
                if mprog:
                    total_chunks = int(mprog.group(1))

        content = str(payload.get("content") or "")
        file_chunks_done += 1

        if not content.strip():
            idx += 1
            continue

        if len(content) <= _MAX_EXTRACT_CHARS:
            running_answer = await _sliding_window_qa_step(
                llm,
                system_prompt,
                question,
                running_answer,
                idx,
                total_chunks,
                content,
                scope_note="",
            )
        else:
            step = max(_MAX_EXTRACT_CHARS - _EXTRACT_OVERLAP, 1)
            fragments = [
                content[i : i + _MAX_EXTRACT_CHARS]
                for i in range(0, len(content), step)
            ]
            nfrag = len(fragments)
            for fi, fr in enumerate(fragments):
                scope = (
                    f"; segment {fi + 1} of {nfrag} within logical chunk {idx} "
                    f"(overlapping sub-windows for length)"
                )
                running_answer = await _sliding_window_qa_step(
                    llm,
                    system_prompt,
                    question,
                    running_answer,
                    idx,
                    total_chunks,
                    fr,
                    scope_note=scope,
                )

        idx += 1

    if not running_answer.strip():
        return {
            "success": True,
            "data": {
                "answer": "No relevant information found in the file.",
                "chunks_read": file_chunks_done,
            },
        }

    answer = await _auto_summary(llm, running_answer, question)
    return {"success": True, "data": {"answer": answer, "chunks_read": file_chunks_done}}


# ---------------------------------------------------------------------------
# text_extract (structured multi-file extraction)
# ---------------------------------------------------------------------------


async def _read_entire_text_file_for_extract(
    mcp_client: "MCPClient",
    file_path: str,
) -> Tuple[str, Optional[str]]:
    """Read full file text; if ``read_file`` truncates, assemble via ``read_file_for_qa``."""
    r = await _read_file_lines(mcp_client, file_path, start_line=1)
    if not r.get("success"):
        return "", str(r.get("error") or "read_file failed")
    raw = str(r.get("data", ""))
    body = _extract_content(r)
    if "(truncated" not in raw.lower():
        return body, None

    buf: List[str] = []
    boundaries_1based: Optional[List[int]] = None
    total_chunks = 1
    idx = 1
    while idx <= total_chunks:
        args: Dict[str, Any] = {"file_path": file_path, "chunk_index": idx}
        if boundaries_1based is not None and idx > 1:
            args["start_line"] = int(boundaries_1based[idx - 1])
            args["end_line"] = int(boundaries_1based[idx]) - 1

        res = await mcp_client.call_tool("read_file_for_qa", args)
        if not res.get("success"):
            return "", str(res.get("error") or "read_file_for_qa failed")
        payload = res.get("data")
        if not isinstance(payload, dict):
            return "", "read_file_for_qa returned unexpected payload"
        if idx == 1:
            b = payload.get("chunk_boundaries")
            if isinstance(b, list) and len(b) >= 2:
                boundaries_1based = [int(x) for x in b]
                total_chunks = len(boundaries_1based) - 1
            else:
                progress = str(payload.get("progress") or "")
                mprog = re.search(r"/(\d+)\s*chunks", progress)
                if mprog:
                    total_chunks = int(mprog.group(1))
        buf.append(str(payload.get("content") or ""))
        idx += 1
    return "".join(buf), None


def _split_text_extract_halo(
    content: str,
    core_window: int,
    overlap: int,
    max_chunks: int,
) -> List[str]:
    total = len(content)
    if total <= core_window:
        return [content]
    chunks: List[str] = []
    start = 0
    while start < total and len(chunks) < max_chunks:
        lo = max(0, start - overlap)
        hi = min(total, start + core_window + overlap)
        chunks.append(content[lo:hi])
        start += core_window
    return chunks


def _parse_text_extract_json(answer: str) -> Tuple[Optional[Any], Optional[str]]:
    text = (answer or "").strip()
    if not text:
        return None, "empty model output"
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, str(e)


async def text_extract(
    mcp_client: "MCPClient",
    llm: "BaseLLM",
    file_paths: List[str],
    task: str,
    *,
    core_window: int = _DEFAULT_TEXT_EXTRACT_CORE_WINDOW,
    halo: int = _DEFAULT_TEXT_EXTRACT_HALO,
    max_chunks_per_doc: int = _DEFAULT_TEXT_EXTRACT_MAX_CHUNKS_PER_DOC,
    max_llm_concurrency: int = _DEFAULT_TEXT_EXTRACT_LLM_CONCURRENCY,
) -> Dict[str, Any]:
    """Structured multi-dimensional extraction over one or more text files (JSON output per chunk).

    Reads each file via MCP (with ``read_file_for_qa`` fallback when ``read_file`` truncates),
    splits large bodies into overlapping windows, and runs the text_extract prompts with bounded
    parallel LLM calls. Results are ordered by (file index, chunk index).
    """
    try:
        task_t = (task or "").strip()
        if not task_t:
            return {"success": False, "data": None, "error": "task must be a non-empty string."}
        paths = [p for p in (file_paths or []) if isinstance(p, str) and p.strip()]
        if not paths:
            return {"success": False, "data": None, "error": "file_paths must contain at least one path."}

        system_prompt = load_prompt_template(str(_TEXT_EXTRACT_SYSTEM_PROMPT_PATH))
        user_tpl = Template(load_prompt_template(str(_TEXT_EXTRACT_USER_MESSAGE_PATH)))

        llm_sem = asyncio.Semaphore(max(1, int(max_llm_concurrency)))

        async def process_chunk(fp: str, chunk: str, chunk_j: int) -> Dict[str, Any]:
            async with llm_sem:
                user_message = user_tpl.safe_substitute(content=chunk, task=task_t)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                try:
                    resp = await llm.generate(messages=messages, tools=None)
                    raw = (resp.get("text_content") or "").strip()
                except Exception as e:
                    return {
                        "file_path": fp,
                        "chunk_index": chunk_j,
                        "task": task_t,
                        "success": False,
                        "error": f"LLM error: {e}",
                    }
                parsed, perr = _parse_text_extract_json(raw)
                meta = {
                    "chars_in_chunk": len(chunk),
                    "lines_in_chunk": len(chunk.splitlines()),
                }
                if perr:
                    meta["json_parse_error"] = perr
                return {
                    "file_path": fp,
                    "chunk_index": chunk_j,
                    "task": task_t,
                    "success": True,
                    "raw_answer": raw,
                    "dimensions": parsed,
                    "metadata": meta,
                }

        async def process_one_doc(doc_order: int, fp: str) -> List[Tuple[int, int, Dict[str, Any]]]:
            suffix = Path(fp).suffix.lower()
            if suffix in BLOCKED_TEXT_SUFFIXES:
                return [
                    (
                        doc_order,
                        0,
                        {
                            "file_path": fp,
                            "chunk_index": 0,
                            "task": task_t,
                            "success": False,
                            "error": (
                                f"Unsupported file type '{suffix}' for text_extract "
                                "(binary/media). Use read_file or a text workflow."
                            ),
                        },
                    )
                ]
            text_body, read_err = await _read_entire_text_file_for_extract(mcp_client, fp)
            if read_err:
                return [
                    (
                        doc_order,
                        0,
                        {
                            "file_path": fp,
                            "chunk_index": 0,
                            "task": task_t,
                            "success": False,
                            "error": read_err,
                        },
                    )
                ]
            text_body = _sanitize_text(text_body)
            if not text_body.strip():
                return [
                    (
                        doc_order,
                        0,
                        {
                            "file_path": fp,
                            "chunk_index": 0,
                            "task": task_t,
                            "success": False,
                            "error": "File is empty or could not be read as text.",
                        },
                    )
                ]
            chunks = _split_text_extract_halo(
                text_body, core_window, halo, max_chunks_per_doc
            )
            outs = await asyncio.gather(
                *(process_chunk(fp, c, j) for j, c in enumerate(chunks))
            )
            return [(doc_order, j, r) for j, r in enumerate(outs)]

        doc_results = await asyncio.gather(
            *(process_one_doc(i, p) for i, p in enumerate(paths))
        )
        flat = [item for sub in doc_results for item in sub]
        flat.sort(key=lambda x: (x[0], x[1]))
        results = [r for _, _, r in flat]
        return {"success": True, "data": {"results": results, "task": task_t}}
    except Exception as e:
        logger.exception("text_extract failed")
        return {"success": False, "data": None, "error": f"text_extract failed: {e}"}


# ---------------------------------------------------------------------------
# text_qa — rich format helper
# ---------------------------------------------------------------------------

async def _text_qa_rich_format(
    mcp_client: "MCPClient",
    llm: "BaseLLM",
    file_path: str,
    question: str,
    system_prompt: str,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> Dict[str, Any]:
    """QA over rich-format files (PDF, DOCX, PPTX, XLSX, CSV, HTML).

    MCP ``read_file`` extracts text from these formats via ``rich_file_reader``.
    We read the full extracted text, chunk it by character count, and run
    sliding-window QA — same logic as the plain-text path but without
    relying on ``read_file_for_qa`` (which does raw UTF-8 I/O).
    """
    result = await _read_file_lines(mcp_client, file_path, start_line=1)
    if not result.get("success"):
        return {
            "success": False,
            "data": None,
            "error": f"Cannot read file: {result.get('error', 'Unknown')}",
        }

    content = _extract_content(result)
    if not content.strip():
        return {
            "success": True,
            "data": {"answer": "The file is empty or could not be extracted.", "chunks_read": 0},
        }

    # If the whole extracted text fits in one chunk, single-round QA
    if len(content) <= max_chunk_chars:
        qa_result = await _single_round_qa(llm, system_prompt, question, content, file_path)
        if qa_result.get("success") and isinstance(qa_result.get("data"), dict):
            answer = qa_result["data"].get("answer", "")
            qa_result["data"]["answer"] = await _auto_summary(llm, answer, question)
        return qa_result

    # Large extracted text → character-based sliding window
    chunks: List[str] = []
    for i in range(0, len(content), max_chunk_chars):
        chunks.append(content[i : i + max_chunk_chars])

    total_chunks = len(chunks)
    running_answer = ""
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        running_answer = await _sliding_window_qa_step(
            llm, system_prompt, question, running_answer,
            idx + 1, total_chunks, chunk,
            scope_note=f" (chars {idx * max_chunk_chars}–{idx * max_chunk_chars + len(chunk)}, {file_path})",
        )

    if not running_answer.strip():
        return {
            "success": True,
            "data": {"answer": "No relevant information found in the file.", "chunks_read": total_chunks},
        }

    answer = await _auto_summary(llm, running_answer, question)
    return {"success": True, "data": {"answer": answer, "chunks_read": total_chunks}}


# ---------------------------------------------------------------------------
# text_qa
# ---------------------------------------------------------------------------

async def text_qa(
    mcp_client: "MCPClient",
    llm: "BaseLLM",
    file_path: str,
    question: str,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
) -> Dict[str, Any]:
    """Text file Q&A with sliding-window processing.

    Single view of the file → one QA call. Large files → sequential chunks; each
    call refines a cumulative answer; the **last** model response is returned
    (then optional length-based auto-summary). All chunks are processed.
    """
    try:
        # ── Input safety ──────────────────────────────────────────────────
        suffix = Path(file_path).suffix.lower()
        if suffix in BLOCKED_TEXT_SUFFIXES:
            return {
                "success": False,
                "data": None,
                "error": (
                    f"text_qa does not support binary/media files (got '{suffix}'). "
                    "Use read_file for raw access, or image_qa for images."
                ),
            }

        system_prompt = load_prompt_template(str(_TEXT_QA_SYSTEM_PROMPT_PATH))

        # ── Rich format path (PDF, DOCX, PPTX, XLSX, CSV, HTML) ─────────
        # MCP read_file extracts text from these formats. read_file_for_qa
        # does raw UTF-8 I/O and cannot handle them, so we always go through
        # read_file and chunk the extracted text ourselves.
        if suffix in RICH_TEXT_SUFFIXES:
            return await _text_qa_rich_format(
                mcp_client, llm, file_path, question, system_prompt, max_chunk_chars,
            )

        # ── Phase 0: probe file size ──────────────────────────────────────
        probe = await _read_file_lines(mcp_client, file_path, start_line=1, end_line=50)
        if not probe.get("success"):
            return {
                "success": False,
                "data": None,
                "error": f"Cannot read file: {probe.get('error', 'Unknown')}",
            }

        probe_content = _extract_content(probe)
        raw_probe = probe.get("data", "")
        probe_truncated = "(truncated" in str(raw_probe).lower()

        total_lines = _parse_total_lines_from_read_file_payload(raw_probe)
        if total_lines == 0:
            # Without "Total lines in file: N", line-based chunking is unsafe (empty chunk list or
            # underestimating size). Delegate to read_file_for_qa which scans the full file once.
            logger.warning(
                "text_qa: could not parse total line count from read_file payload — "
                "using read_file_for_qa path for %s",
                file_path,
            )
            return await _text_qa_via_read_file_for_qa_chunks(
                mcp_client, llm, file_path, question, system_prompt,
            )

        # Probe hit read_file 40k cap (common for JSONL: line 1 is giant tool_schema). Line-based
        # chunking using total_lines alone is insufficient — use read_file_for_qa (~96k chunks).
        if probe_truncated:
            logger.info("text_qa: probe truncated → read_file_for_qa path for %s", file_path)
            return await _text_qa_via_read_file_for_qa_chunks(
                mcp_client, llm, file_path, question, system_prompt,
            )

        probe_lines = probe_content.count("\n") + 1
        avg_chars_per_line = max(len(probe_content) / max(probe_lines, 1), 40)
        lines_per_chunk = max(int(max_chunk_chars / avg_chars_per_line), 20)

        chunks = _compute_line_chunks(total_lines, lines_per_chunk)
        total_chunks = len(chunks)
        if total_chunks == 0:
            return {
                "success": True,
                "data": {"answer": "The file is empty.", "chunks_read": 0},
            }

        logger.info(
            "text_qa: file=%s total_lines=%d avg_cpl=%.0f lines/chunk=%d chunks=%d",
            file_path, total_lines, avg_chars_per_line, lines_per_chunk, total_chunks,
        )

        # ── Small-file fast path ──────────────────────────────────────────
        if total_chunks == 1:
            result = await _read_file_lines(mcp_client, file_path, start_line=1)
            raw_full = str(result.get("data", ""))
            if "(truncated" in raw_full.lower():
                logger.info("text_qa: full read still truncated → read_file_for_qa path for %s", file_path)
                return await _text_qa_via_read_file_for_qa_chunks(
                    mcp_client, llm, file_path, question, system_prompt,
                )
            content = _extract_content(result)
            qa_result = await _single_round_qa(llm, system_prompt, question, content, file_path)
            if qa_result.get("success") and isinstance(qa_result.get("data"), dict):
                answer = qa_result["data"].get("answer", "")
                qa_result["data"]["answer"] = await _auto_summary(llm, answer, question)
            return qa_result

        # ── Sliding window: each line-range chunk updates cumulative answer ─
        running_answer = ""
        chunks_read = 0

        for idx, (start, end) in enumerate(chunks):
            result = await _read_file_lines(
                mcp_client, file_path, start_line=start, end_line=end
            )
            raw_chunk = str(result.get("data", ""))
            if "(truncated" in raw_chunk.lower():
                logger.info(
                    "text_qa: read_file hit 40k cap mid-chunk (lines %s–%s) → read_file_for_qa path for %s",
                    start,
                    end,
                    file_path,
                )
                return await _text_qa_via_read_file_for_qa_chunks(
                    mcp_client, llm, file_path, question, system_prompt,
                )
            content = _extract_content(result)
            chunks_read += 1
            if not content.strip():
                continue

            scope_note = f" (file lines {start}–{end})"
            running_answer = await _sliding_window_qa_step(
                llm,
                system_prompt,
                question,
                running_answer,
                idx + 1,
                total_chunks,
                content,
                scope_note=scope_note,
            )

        if not running_answer.strip():
            return {
                "success": True,
                "data": {
                    "answer": "No relevant information found in the file.",
                    "chunks_read": chunks_read,
                },
            }

        answer = await _auto_summary(llm, running_answer, question)

        return {
            "success": True,
            "data": {"answer": answer, "chunks_read": chunks_read},
        }

    except Exception as e:
        logger.exception("text_qa failed")
        return {"success": False, "data": None, "error": f"text_qa failed: {e}"}


# ---------------------------------------------------------------------------
# text_qa internal helpers
# ---------------------------------------------------------------------------

async def _sliding_window_qa_step(
    llm: "BaseLLM",
    system_prompt: str,
    question: str,
    prior_cumulative: str,
    chunk_index: int,
    total_chunks: int,
    chunk_content: str,
    scope_note: str = "",
) -> str:
    """One step of sliding-window QA: merge *prior_cumulative* with *chunk_content*."""
    user_parts: List[str] = [f"Question: {question}\n"]
    if prior_cumulative.strip():
        user_parts.append(
            "--- Accumulated answer from prior chunks "
            "(same question; keep all necessary facts — do not discard) ---\n"
            f"{prior_cumulative.strip()}\n"
        )
    user_parts.append(
        f"--- Current chunk to analyze: chunk {chunk_index} of {total_chunks}"
        f"{scope_note} ---\n"
        f"{chunk_content}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    try:
        resp = await llm.generate(messages=messages)
        return resp.get("text_content", "").strip()
    except Exception as e:
        logger.warning(
            "Sliding-window QA step failed (chunk %s/%s): %s",
            chunk_index,
            total_chunks,
            e,
        )
        return prior_cumulative


async def _single_round_qa(
    llm: "BaseLLM",
    system_prompt: str,
    question: str,
    content: str,
    file_path: str,
) -> Dict[str, Any]:
    try:
        answer = await _sliding_window_qa_step(
            llm,
            system_prompt,
            question,
            prior_cumulative="",
            chunk_index=1,
            total_chunks=1,
            chunk_content=content,
            scope_note=f" ({file_path}; full file in one pass)",
        )
    except Exception as e:
        return {"success": False, "data": None, "error": f"LLM call failed: {e}"}
    if not answer:
        return {"success": False, "data": None, "error": "LLM returned an empty answer."}
    return {"success": True, "data": {"answer": answer, "chunks_read": 1}}


# ---------------------------------------------------------------------------
# image_qa
# ---------------------------------------------------------------------------

def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


async def image_qa(
    mcp_client: "MCPClient",
    llm: "BaseLLM",
    file_path: str,
    question: str,
) -> Dict[str, Any]:
    """Image file Q&A using VLM (single-round).

    Accepts workspace-relative paths only; remote URLs are rejected.
    Rejects non-image files (PDF, documents, etc.) with clear error messages.
    Auto-summarizes if the VLM response is excessively long.
    """
    try:
        # ── URL handling ──────────────────────────────────────────────────
        actual_path = file_path
        if _is_url(file_path):
            return {
                "success": False,
                "data": None,
                "error": (
                    "image_qa only accepts workspace-relative paths. This deployment "
                    "registers no network-fetch tool, so remote URLs cannot be retrieved. "
                    "Save the image into the workspace first, then call image_qa on it."
                ),
            }

        # ── Extension safety ──────────────────────────────────────────────
        suffix = Path(actual_path).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            hint = "Use text_qa for text files, or read_file for raw access."
            if suffix == ".pdf":
                hint = "PDF files cannot be processed as images. Use text_qa or read_file instead."
            elif suffix in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
                hint = "Office documents are not images. Convert to text first, then use text_qa."
            return {
                "success": False,
                "data": None,
                "error": (
                    f"image_qa only accepts image files "
                    f"({', '.join(sorted(ALLOWED_IMAGE_SUFFIXES))}), "
                    f"but got '{suffix or 'no extension'}'. {hint}"
                ),
            }

        # ── Read image via MCP ────────────────────────────────────────────
        result = await mcp_client.call_tool("read_image_for_qa", {"file_path": actual_path})
        if not result.get("success"):
            return {
                "success": False,
                "data": None,
                "error": f"Failed to read image: {result.get('error', 'Unknown error')}",
            }

        image_data = result.get("data", {})
        data_uri = image_data.get("data_uri", "")
        mime_type = image_data.get("mime_type", "")

        if not data_uri:
            return {"success": False, "data": None, "error": "No image data returned from read_image_for_qa"}

        if mime_type and str(mime_type).lower() not in ALLOWED_IMAGE_MIME_TYPES:
            return {
                "success": False,
                "data": None,
                "error": (
                    f"Unexpected MIME type '{mime_type}'. "
                    f"Expected one of: {', '.join(sorted(ALLOWED_IMAGE_MIME_TYPES))}."
                ),
            }

        # ── VLM call ──────────────────────────────────────────────────────
        system_prompt = load_prompt_template(str(_IMAGE_QA_SYSTEM_PROMPT_PATH))
        user_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await llm.generate(messages=messages, tools=None)
            answer = response.get("text_content", "").strip()
        except Exception as e:
            return {"success": False, "data": None, "error": f"VLM call failed: {e}"}

        # ── Auto-summary for excessively long VLM output ──────────────────
        if len(answer) > MAX_OUTPUT_CHARS:
            answer = await _auto_summary(llm, answer, question)

        return {"success": True, "data": {"answer": answer}}

    except Exception as e:
        return {"success": False, "data": None, "error": f"image_qa failed: {e}"}
