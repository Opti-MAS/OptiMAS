"""LLM utility functions and error handling"""
import asyncio
import logging
from typing import Tuple, Any

from ..utils.exceptions import ContextOverflowError

logger = logging.getLogger(__name__)

# Avoid overly broad substrings (e.g. bare "max_tokens" can appear in unrelated validation errors).
_CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "reduce the length",
    "token limit",
    "request too large",
    "prompt is too long",
    "input too long",
    "too many tokens",
)


def is_context_overflow(e: Exception) -> bool:
    """Check whether *e* signals that the prompt exceeded the model's context window."""
    msg = str(e).lower()
    return any(p in msg for p in _CONTEXT_OVERFLOW_PATTERNS)


_MODEL_NOT_FOUND_PATTERNS = (
    "model_not_found",
    "model not found",
    "the model does not exist",
    "no such model",
)


def is_model_not_found(e: Exception) -> bool:
    """Check whether *e* indicates the requested model is unavailable / not found."""
    msg = str(e).lower()
    return any(p in msg for p in _MODEL_NOT_FOUND_PATTERNS)


def is_server_side_error(e: Exception) -> bool:
    """Check whether *e* is a server/infrastructure error worth trying a fallback model.

    Returns True for: model_not_found, 5xx status, rate limits, connection/timeout
    errors, and other non-content failures.  Returns False for content/validation
    errors (bad JSON, duplicate tool calls, empty response) where switching models
    would not help.
    """
    name = type(e).__name__
    msg = str(e).lower()

    if is_model_not_found(e):
        return True

    if any(code in msg for code in ("status code: 5", "500", "502", "503", "504")):
        return True

    if "rate limit" in msg or "429" in msg or "too many requests" in msg:
        return True

    if any(kw in name.lower() for kw in ("connection", "timeout")):
        return True
    if any(kw in msg for kw in ("connection", "timeout", "timed out", "connect error")):
        return True

    if "internalservererror" in name.lower() or "serviceunavailable" in name.lower():
        return True

    return False


async def handle_exception(
    e: Exception,
    response: Any = None,
) -> Tuple[str, str, float]:
    """Unified error handling for LLM API calls.

    Returns:
        (error_message_for_llm, detailed_error_for_log, attempt_increment)

    Raises:
        ContextOverflowError: immediately, when the error indicates
            context-length overflow (retrying won't help without truncation).
    """
    error_type_name = type(e).__name__
    err_str = str(e).strip() if str(e).strip() else "(empty)"

    error_info_parts = [f"Exception type: {error_type_name}", f"Error message: {err_str}"]
    if hasattr(e, 'code'):
        error_info_parts.append(f"Error code: {e.code}")
    if hasattr(e, 'status_code'):
        error_info_parts.append(f"Status code: {e.status_code}")
    detailed_error = " | ".join(error_info_parts)

    # Context overflow → NOT retryable.  Raise immediately so the agent
    # layer can compact/truncate before the next attempt.
    if is_context_overflow(e):
        raise ContextOverflowError(
            f"Context length exceeded: {err_str}"
        ) from e

    retry_increment = 1.0

    if 'missing a thought_signature' in err_str.lower():
        error_message = ""
        sleep_seconds = 5.0
        retry_increment = 0.1
    elif 'response message has neither tool_calls nor content' in err_str.lower():
        error_message = "Response is empty, generate again."
        sleep_seconds = 5.0
        retry_increment = 0.2
    elif error_type_name == "ResponseValidationError" or (hasattr(e, 'response') and hasattr(e, 'message')):
        error_message = f"Response is not valid, generate again. Here is the error message: {err_str}"
        sleep_seconds = 1.0
    elif "rate limit" in err_str.lower() or "429" in err_str:
        error_message = ""
        sleep_seconds = 5.0
    elif "Connection" in error_type_name or "connection" in err_str.lower():
        error_message = ""
        sleep_seconds = 5.0
    elif "timeout" in err_str.lower() or "Timeout" in error_type_name:
        error_message = "Timeout occurred. Generate a shorter response or break down the task into smaller steps, one step at a time."
        sleep_seconds = 5.0
    else:
        error_message = err_str
        sleep_seconds = 2.0

    await asyncio.sleep(sleep_seconds)
    return (error_message, detailed_error, retry_increment)
