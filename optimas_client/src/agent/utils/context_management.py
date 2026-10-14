"""Conversation history and context compression helpers.

Provides:
  - ``safe_add_user_prompt`` / ``stamp_trajectory_timestamp`` for trajectory analysis;
  - History splitting, formatting, and optional ``compress_conversation_with_llm``
    aligned with the context-compression system prompt, enabling semantic-preserving
    compression before tail-drop in the Agent layer.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...llm.base import BaseLLM

TRAJECTORY_TIMESTAMP_KEY = "_trajectory_timestamp"

_CONTEXT_COMPRESSION_PROMPT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "prompts"
    / "internal_ai_tools"
    / "context_compression_system_prompt.txt"
)


def stamp_trajectory_timestamp(message: Dict[str, Any]) -> None:
    """Set per-message UTC ISO timestamp for offline trajectory analysis."""
    message[TRAJECTORY_TIMESTAMP_KEY] = datetime.now(timezone.utc).isoformat()


def safe_add_user_prompt(conversation_history: List[Dict[str, Any]], prompt: str) -> None:
    """Safely append a user prompt to conversation history.

    If the last message is already a user message, the prompt is concatenated
    with a separator; otherwise a new user message is appended.
    """
    if conversation_history and conversation_history[-1].get("role") == "user":
        conversation_history[-1]["content"] += "\n\n---\n\n" + prompt
        stamp_trajectory_timestamp(conversation_history[-1])
    else:
        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        stamp_trajectory_timestamp(msg)
        conversation_history.append(msg)


def _deepcopy_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return deepcopy(messages)


def estimate_dict_messages_token_budget(messages: List[Dict[str, Any]]) -> int:
    """Heuristic token upper bound: chars/3 + 4/msg (conservative estimate)."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text")
                    if isinstance(t, str):
                        total_chars += len(t)
        if msg.get("role") == "assistant":
            rc = msg.get("reasoning_content") or msg.get("reasoning_summary")
            if isinstance(rc, str):
                total_chars += len(rc)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if isinstance(fn, dict):
                    total_chars += len(str(fn.get("name", ""))) + len(
                        str(fn.get("arguments", ""))
                    )
                else:
                    total_chars += len(str(tc))
    return total_chars // 3 + 4 * len(messages)


def split_history_for_llm_compression(
    messages: List[Dict[str, Any]],
) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Split history into (preamble, compressible, last_round).

    - preamble: all messages before the first assistant message;
    - compressible: from the first assistant to just before the last assistant;
    - last_round: from the last assistant message to the end.

    Returns ``None`` if there is no compressible segment.
    """
    preamble_end: Optional[int] = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            preamble_end = i
            break
    if preamble_end is None:
        return None

    last_assistant_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx <= preamble_end:
        return None

    preamble = messages[:preamble_end]
    compressible = messages[preamble_end:last_assistant_idx]
    last_round = messages[last_assistant_idx:]
    return preamble, compressible, last_round


def extract_task_context_from_preamble(
    preamble: List[Dict[str, Any]],
    max_chars: int = 500,
) -> str:
    """Extract the first user message from preamble as task context for the compression prompt."""
    for msg in preamble:
        if msg.get("role") != "user":
            continue
        raw = msg.get("content")
        if not isinstance(raw, str):
            raw = str(raw)
        excerpt = raw[:max_chars] + ("..." if len(raw) > max_chars else "")
        return f"Task assignment:\n{excerpt}\n\n"
    return ""


def format_compressible_messages_for_llm(
    compressible: List[Dict[str, Any]],
    *,
    tool_body_max: int = 2000,
) -> str:
    """Flatten compressible messages into a single text block for the compression LLM.

    Tool outputs longer than *tool_body_max* are truncated.
    """
    text_parts: List[str] = []
    for msg in compressible:
        role = msg.get("role", "")
        if role == "assistant":
            parts: List[str] = []
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                if isinstance(fn, dict):
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                else:
                    name, args = str(tc), ""
                parts.append(f"[Tool call: {name}({args})]")
            text_parts.append(
                f"assistant: {' '.join(parts) if parts else '(no text)'}"
            )
        elif role == "tool":
            body = msg.get("content", "")
            if isinstance(body, str) and len(body) > tool_body_max:
                body = body[:tool_body_max] + "... (truncated)"
            label = msg.get("name") or msg.get("tool_call_id") or "tool"
            text_parts.append(f"tool({label}): {body}")
        elif role == "user":
            c = msg.get("content")
            text_parts.append(f"user: {c if isinstance(c, str) else str(c)}")
        else:
            c = msg.get("content", "")
            text_parts.append(f"{role}: {c if isinstance(c, str) else str(c)}")
    return "\n\n".join(text_parts)


def build_compressed_replacement_history(
    preamble: List[Dict[str, Any]],
    summary_text: str,
    last_round: List[Dict[str, Any]],
    *,
    wrapped_count: int,
) -> List[Dict[str, Any]]:
    """Build the compressed history: preamble + summary user message + last_round."""
    merged = (
        f"[Context compressed — {wrapped_count} messages summarized]\n\n{summary_text}"
    )
    out: List[Dict[str, Any]] = []
    out.extend(_deepcopy_messages(preamble))
    out.append({"role": "user", "content": merged})
    out.extend(_deepcopy_messages(last_round))
    return out


async def compress_conversation_with_llm(
    llm: "BaseLLM",
    messages: List[Dict[str, Any]],
    *,
    tool_body_max: int = 2000,
) -> Optional[List[Dict[str, Any]]]:
    """Run one round of LLM-based context compression.

    Returns a **new** message list on success (the input is not modified).
    Returns ``None`` if there is no compressible segment or the LLM call fails.
    """
    split = split_history_for_llm_compression(messages)
    if split is None:
        return None
    preamble, compressible, last_round = split

    from .prompt_utils import load_prompt_template

    system_prompt = load_prompt_template(str(_CONTEXT_COMPRESSION_PROMPT))
    task_context = extract_task_context_from_preamble(preamble)
    conversation_text = format_compressible_messages_for_llm(
        compressible, tool_body_max=tool_body_max
    )

    req: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{task_context}"
                f"Compress the following conversation ({len(compressible)} messages):\n\n"
                f"{conversation_text}"
            ),
        },
    ]
    try:
        resp = await llm.generate(messages=req, tools=None)
    except Exception:
        return None
    summary = (resp.get("text_content") or "").strip()
    if not summary:
        return None
    return build_compressed_replacement_history(
        preamble, summary, last_round, wrapped_count=len(compressible)
    )


def strip_private_keys_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a shallow copy of messages with underscore-prefixed keys removed."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        out.append({k: v for k, v in m.items() if not str(k).startswith("_")})
    return out


def estimate_public_history_chars(messages: List[Dict[str, Any]]) -> int:
    """Estimate size of public (non-underscore) fields, matching ``BaseAgent._estimate_message_size``."""
    total = 0
    for msg in messages:
        try:
            public = {k: v for k, v in msg.items() if not str(k).startswith("_")}
            total += len(json.dumps(public, ensure_ascii=False))
        except Exception:
            total += len(str(msg))
    return total
