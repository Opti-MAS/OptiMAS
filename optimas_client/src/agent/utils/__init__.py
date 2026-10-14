"""Agent utilities module"""
from .context_management import (
    TRAJECTORY_TIMESTAMP_KEY,
    build_compressed_replacement_history,
    compress_conversation_with_llm,
    estimate_dict_messages_token_budget,
    estimate_public_history_chars,
    extract_task_context_from_preamble,
    format_compressible_messages_for_llm,
    safe_add_user_prompt,
    split_history_for_llm_compression,
    stamp_trajectory_timestamp,
    strip_private_keys_for_api,
)
from .tool_decorator import mcp_tool, collect_tool_schemas
from .prompt_utils import (
    load_prompt_template,
    load_skill,
    format_prompt_template,
    build_system_prompt_with_skills
)

__all__ = [
    "mcp_tool",
    "collect_tool_schemas",
    "load_prompt_template",
    "load_skill",
    "format_prompt_template",
    "build_system_prompt_with_skills",
    "safe_add_user_prompt",
    "stamp_trajectory_timestamp",
    "TRAJECTORY_TIMESTAMP_KEY",
    "split_history_for_llm_compression",
    "extract_task_context_from_preamble",
    "format_compressible_messages_for_llm",
    "build_compressed_replacement_history",
    "compress_conversation_with_llm",
    "estimate_dict_messages_token_budget",
    "estimate_public_history_chars",
    "strip_private_keys_for_api",
]
