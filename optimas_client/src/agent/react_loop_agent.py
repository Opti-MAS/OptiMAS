"""ReAct loop core: LLM turns, tool schema build, prompt assembly, tool batch dispatch, and iteration / context guards.

``BaseAgent`` (``base_react_agent.py``) subclasses this and adds internal AI tools (text_qa, …),
sub-agent delegation, proactive LLM context compression, and tail-drop compaction.
"""
import asyncio
import inspect
import json
import logging
import re
import time
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple

from ..config.loader import ConfigLoader
from ..llm.factory import create_llm
from ..mcp.client import MCPClient
from ..messaging.emitter import MessageEmitter
from ..messaging.types import MessageType
from ..utils.exceptions import LLMError, ContextOverflowError
from .utils.tool_decorator import collect_tool_schemas
from .utils.prompt_utils import (
    build_system_prompt_with_skills,
)
from .utils.context_management import (
    safe_add_user_prompt,
    stamp_trajectory_timestamp,
    compress_conversation_with_llm,
    estimate_dict_messages_token_budget,
)

logger = logging.getLogger(__name__)

_TRUNCATION_GUIDANCE: Dict[str, str] = {
    "read_file": "Use start_line/end_line to fetch small windows, or use text_qa for document understanding.",
    "text_qa": "Ask a more focused question, or split the file into smaller sections.",
    "text_extract": "Reduce the number of files per call or narrow the task scope.",
}


class BaseReactAgent:
    """Core ReAct loop: prompts, tool schemas, ``execute_task``, iteration guards, and tool-call batching.

    Concrete workers subclass :class:`BaseAgent` in ``base_react_agent.py`` for internal tools,
    delegation, and context compression.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        config: Dict[str, Any],
        message_emitter: MessageEmitter,
        recording_config: Optional[Dict[str, Any]] = None,
        global_llm_settings: Optional[Dict[str, Any]] = None,
        global_config: Optional[Dict[str, Any]] = None,
        ancestor_agents: Optional[Set[str]] = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.config = config or {}
        self.global_config = global_config or {}
        self.message_emitter = message_emitter

        # Agent identity (for cycle detection in multi-agent delegation)
        self._agent_name: str = self.config.get("agent_name", "")
        self._ancestor_agents: Set[str] = ancestor_agents or set()

        self.max_iterations = self.config.get("max_iterations", 20)
        self.max_real_iterations = self.config.get("max_real_iterations", 60)

        self.max_consecutive_llm_failures = int(self.config.get("max_consecutive_llm_failures", 3))
        self.max_consecutive_no_tool_calls = int(self.config.get("max_consecutive_no_tool_calls", 3))

        self._context_soft_limit = int(self.config.get("context_soft_limit_chars", 200000))
        self._context_hard_limit = int(self.config.get("context_hard_limit_chars", 240000))
        self._context_must_limit = int(self.config.get("context_must_limit_chars", 260000))

        self._max_real_warn_threshold = int(self.config.get("max_real_iteration_warning_threshold", 5))
        self._max_real_task_done_only_threshold = int(self.config.get("max_real_iteration_task_done_only_threshold", 3))

        self._response_tags: List[str] = list(self.config.get("response_tags", []))
        
        self.available_tool_names: Set[str] = set(self.config.get("available_tools", []))
        self.mcp_tool_names: Set[str] = set()
        
        # Initialize all LLMs
        self._init_large_language_models(recording_config, global_llm_settings)

        # Prompt and Skill config (prefer agent config, fall back to global config)
        global_config_dir = self.global_config.get("_config_dir", ".")
        config_dir = self.config.get("_config_dir", global_config_dir)
        
        # Process prompt_dict (resolve to absolute paths)
        self.prompt_dict: Optional[Dict[str, str]] = self.config.get("prompt_dict")
        if self.prompt_dict:
            new_prompt_dict = {}
            for k, v in self.prompt_dict.items():
                if not Path(v).is_absolute():
                    new_prompt_dict[k] = str((Path(config_dir) / v).resolve())
                else:
                    new_prompt_dict[k] = str(Path(v).resolve())
            self.prompt_dict = new_prompt_dict

        global_skill_raw = ConfigLoader.get_skill_dir(self.global_config)
        if global_skill_raw:
            if not Path(global_skill_raw).is_absolute():
                _gsd = (Path(global_config_dir) / global_skill_raw).absolute()
            else:
                _gsd = Path(global_skill_raw)
            self.global_skill_dir: Optional[str] = str(_gsd) if _gsd.is_dir() else None
        else:
            self.global_skill_dir = None
            
        # Agent-specific skill directory (relative paths resolved against config_dir)
        skill_raw = self.config.get("skill_dir")
        if skill_raw:
            if not Path(skill_raw).is_absolute():
                self.skill_dir: Optional[str] = str((Path(config_dir) / skill_raw).resolve())
            else:
                self.skill_dir = str(Path(skill_raw).resolve())
        else:
            self.skill_dir = self.global_skill_dir
            
        self.system_skills: List[str] = self.config.get("system_skills", [])
        self.optional_skills: List[str] = self.config.get("optional_skills", [])
        self.preload_skills: List[str] = self.config.get("preload_skills", [])
        
        # Conversation history (initialized in execute_task)
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Tool schemas (lazy-loaded, built asynchronously in execute_task)
        self.tool_schemas: Optional[List[Dict[str, Any]]] = None
        
        # Track saved history length for incremental persistence
        self._last_saved_history_length: int = 0

        # Per-ancestor sub-agent store: at most one instance per subagent name.
        # Enables resuming a previously invoked sub-agent with its full conversation context.
        self._subagent_store: Dict[str, Any] = {}

        # Sub-agents this agent is allowed to delegate to via assign_task_to_agent
        _subagent = self.config.get("subagent")
        if _subagent is None:
            self.allowed_subagents: Set[str] = set()
        elif isinstance(_subagent, list):
            self.allowed_subagents = set(str(x) for x in _subagent)
        else:
            self.allowed_subagents = set()

    def _init_large_language_models(self, recording_config: Optional[Dict[str, Any]] = None, global_llm_settings: Optional[Dict[str, Any]] = None) -> None:
        # Create LLM from config
        llm_config = self.config.get("llm_config")
        if not llm_config:
            llm_config = self.global_config.get("llm_config")
            if not llm_config:
                raise ValueError(
                    "No LLM config: set ``llm_config`` on this agent or under global config."
                )
        self.llm = create_llm(
            config=llm_config,
            recording_config=recording_config,
            global_llm_settings=global_llm_settings
        )
        # Internal tools can use a separate LLM; defaults to the main LLM
        internal_llm_config = self.config.get("internal_tools_llm_config")
        if not internal_llm_config:
            internal_llm_config = self.global_config.get("internal_tools_llm_config")
            if not internal_llm_config:
                internal_llm_config = llm_config.copy()
        self.internal_tools_llm = create_llm(
            config=internal_llm_config,
            recording_config=recording_config,
            global_llm_settings=global_llm_settings
        )

        # image_qa can use a dedicated VLM; falls back to the internal-tools LLM
        # config rather than any hardcoded provider endpoint.
        vlm_config = self.config.get("vlm_config") or self.global_config.get("vlm_config")
        if not vlm_config:
            vlm_config = internal_llm_config.copy()
        self.vlm = create_llm(
            config=vlm_config,
            recording_config=recording_config,
            global_llm_settings=global_llm_settings
        )

    # =====================================================================
    # Context size (soft/hard limits; used by guards and compaction hooks)
    # =====================================================================
    def _estimate_message_size(self, msg: Dict[str, Any]) -> int:
        """Roughly estimate payload size for context compaction.

        Excludes keys starting with ``_`` — those are dev-only (e.g. timing for JSONL
        trajectories) and are stripped before any LLM API call by
        ``LLMClient._clean_messages_for_api``; they must not inflate soft/hard limits.
        """
        try:
            public = {k: v for k, v in msg.items() if not str(k).startswith("_")}
            return len(json.dumps(public, ensure_ascii=False))
        except Exception:
            return len(str(msg))

    def _estimate_conversation_history_size(self) -> int:
        """Estimate current conversation history payload size in characters."""
        return sum(self._estimate_message_size(m) for m in self.conversation_history)

    def _compact_conversation_history_for_context(self, target_chars: Optional[int] = None,) -> bool:
        """
        Compact conversation history: keep system message + most-recent tail
        within *target_chars*. Returns True if compaction actually happened.
        """
        if target_chars is None:
            target_chars = int(self._context_hard_limit * 0.7)

        if not self.conversation_history:
            return False

        start_idx = next((i for i, message in enumerate(self.conversation_history)
                          if message.get("role") == "assistant"), len(self.conversation_history))
        preamble = self.conversation_history[:start_idx]
        tail = self.conversation_history[start_idx:]
        groups: List[List[Dict[str, Any]]] = []
        for message in tail:
            if message.get("role") == "assistant" or not groups:
                groups.append([])
            groups[-1].append(message)
        kept_groups = []
        used = sum(self._estimate_message_size(message) for message in preamble)
        for group in reversed(groups):
            size = sum(self._estimate_message_size(message) for message in group)
            if kept_groups and used + size > target_chars:
                break
            kept_groups.append(group)
            used += size
        kept_tail = [message for group in reversed(kept_groups) for message in group]
        if len(kept_tail) >= len(tail):
            return False

        new_history: List[Dict[str, Any]] = list(preamble)
        new_history.extend(kept_tail)
        self.conversation_history = new_history
        self._last_saved_history_length = min(self._last_saved_history_length, len(self.conversation_history))

        safe_add_user_prompt(
            self.conversation_history,
            "Context window was compacted automatically. Continue from current state; avoid rereading large files unless necessary."
        )
        return True

    
    # =====================================================================
    # Force phase transition
    # =====================================================================
    _COMPRESSED_MSG_PREFIX = "[Context compressed"

    async def _emit_phase_transition(self, summary: str, remaining: str) -> None:
        """Emit phase boundary events and persist the full refreshed history."""
        if len(summary) > 8000:
            summary = summary[:8000] + "\n...[truncated for phase_summary event]"

        await self.message_emitter.emit_phase(
            progress_report=summary,
            remaining_tasks=remaining,
            iteration=self.iteration,
        )
        self.message_emitter.increment_phase()

        if self.tool_schemas:
            await self.message_emitter.emit(
                message_type=MessageType.TOOL_SCHEMA,
                metadata={"tool_schemas": self.tool_schemas},
            )

        self._last_saved_history_length = 0
        ok = await self.message_emitter.emit(
            message_type=MessageType.CONVERSATION_HISTORY,
            metadata={
                "iteration": self.iteration,
                "real_iteration": self.real_iteration,
                "history": self.conversation_history,
            },
        )
        if ok:
            self._last_saved_history_length = len(self.conversation_history)

    def _find_compressed_summary(self) -> str:
        """Return the content of the ``[Context compressed …]`` user message, or a fallback."""
        for m in self.conversation_history:
            if (
                m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith(self._COMPRESSED_MSG_PREFIX)
            ):
                return m["content"]
        return "[Context compressed]"

    def _append_user_message_always(self, content: str) -> None:
        """Append a user message without merging into the previous user turn.

        Used after LLM context compression so the bridge notice never corrupts the
        structured ``[Context compressed — …]`` summary block.
        """
        msg: Dict[str, Any] = {"role": "user", "content": content}
        stamp_trajectory_timestamp(msg)
        self.conversation_history.append(msg)

    async def _apply_llm_compressed_history(self, new_hist: List[Dict[str, Any]], est_before: int, trigger: str,) -> None:
        """Replace conversation_history with LLM-compressed version and emit phase transition."""
        self.conversation_history = new_hist

        for m in self.conversation_history:
            if (
                m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].startswith(self._COMPRESSED_MSG_PREFIX)
            ):
                stamp_trajectory_timestamp(m)
                break

        self._append_user_message_always(
            "[System] The conversation was **automatically compressed** using an LLM summary "
            "(see the structured summary user message above). Treat earlier raw tool I/O as "
            "absorbed into that summary unless you re-invoke tools."
        )

        await self._emit_phase_transition(
            summary=self._find_compressed_summary(),
            remaining=(
                "Continue the same task. Earlier turns were summarized in-place (LLM compression); "
                "verbatim pre-summary tool traces are no longer in context."
            ),
        )

        est_after = estimate_dict_messages_token_budget(self.conversation_history)
        logger.info(
            "LLM context compression applied (trigger=%s iter=%s before=%s after=%s)",
            trigger, self.iteration, est_before, est_after,
        )

    async def _force_compress_context(self, *, skip_threshold: bool = False) -> bool:
        """Force LLM summary compression, with tail-drop fallback.

        Args:
            skip_threshold: If True, skip the must-limit size check and compress
                unconditionally.  Used by the ContextOverflowError handler where
                the LLM has already rejected the request.

        Returns True if any form of compression was applied.
        """
        if not self.conversation_history:
            return False
        if not skip_threshold:
            if self._estimate_conversation_history_size() <= self._context_must_limit:
                return False

        trigger = "overflow" if skip_threshold else "must_limit"
        est_before = estimate_dict_messages_token_budget(self.conversation_history)

        try:
            new_hist = await compress_conversation_with_llm(
                self.llm, self.conversation_history,
            )
        except Exception as exc:
            logger.warning("LLM compression failed (%s): %s; falling back to tail-drop.", trigger, exc)
            new_hist = None

        if new_hist:
            await self._apply_llm_compressed_history(new_hist, est_before, trigger)
            return True

        target = int(self._context_hard_limit * 0.7)
        compacted = self._compact_conversation_history_for_context(target_chars=target)
        if compacted:
            logger.warning("Fell back to tail-drop compact (%s, target=%s chars).", trigger, target)
        return compacted

    # =====================================================================
    # Phase / context guards & tool exposure (policy + schema narrowing)
    # =====================================================================
    _CONTEXT_SOFT_WARN_PROMPT = (
        "[CONTEXT_SOFT_WARNING] The conversation context is approaching its size limit. "
        "Please wrap up your current work and call phase_done_and_summarize soon to "
        "preserve your progress before the context is forcibly compacted."
    )
    _CONTEXT_HARD_GUARD_PROMPT = (
        "[CONTEXT_HARD_LIMIT] Your context has exceeded the hard limit. "
        "All tools except `phase_done_and_summarize` and `task_done` have been disabled. "
        "You MUST call one of these two tools NOW:\n"
        "- `phase_done_and_summarize`: summarize your completed progress and remaining tasks, "
        "then continue in the next phase with a clean context.\n"
        "- `task_done`: if your task is complete or you have a best-effort answer, submit it.\n"
        "Do NOT attempt to call any other tool — it will not be available."
    )


    def _check_context_soft_warning(self) -> bool:
        """Inject a warning nudge when context reaches the soft limit (no tool restriction)."""
        ctx_size = self._estimate_conversation_history_size()
        if ctx_size < max(self._context_soft_limit, 200000):
            return False
        self._inject_user_nudge_unless_recent_contains(
            "[CONTEXT_SOFT_WARNING]", self._CONTEXT_SOFT_WARN_PROMPT
        )
        return True

    # =====================================================================
    # Hard context limit error 
    # =====================================================================
    def _check_context_hard_limit(self) -> bool:
        """Check if context exceeds the hard limit.

        If so, restrict tools to ``{phase_done_and_summarize, task_done}`` and inject
        a stern guard prompt.  Returns True when the hard limit was exceeded.
        The caller is responsible for applying the returned tool restriction.
        """
        return self._estimate_conversation_history_size() > self._context_hard_limit

    def _get_hard_limit_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Return tool schemas restricted to phase_done_and_summarize + task_done."""
        return self._expose_only_tool_names({"phase_done_and_summarize", "task_done"})

    def _inject_hard_limit_guard_if_needed(self) -> None:
        """Inject hard-limit guard prompt unless already present in recent messages."""
        self._inject_user_nudge_unless_recent_contains(
            "[CONTEXT_HARD_LIMIT]", self._CONTEXT_HARD_GUARD_PROMPT
        )

    def _should_force_phase_done(self, reason_out: Optional[List[str]] = None) -> bool:
        """Check non-context conditions that force tool restriction to phase_done.

        Currently: consecutive no-tool-call streak.
        """
        triggered = False
        if getattr(self, "_consecutive_no_tool_calls", 0) >= self.max_consecutive_no_tool_calls:
            triggered = True
            if reason_out is not None:
                reason_out.append(f"consecutive_no_tool_calls ({self._consecutive_no_tool_calls})")
        return triggered

    # --- Tool exposure: generic schema filter + small policy-specific name sets (compose, do not merge). ---
    def _expose_only_tool_names(self, allowed: Set[str]) -> Optional[List[Dict[str, Any]]]:
        """Return schemas from ``self.tool_schemas`` whose ``name`` is in *allowed* (stable order)."""
        if not self.tool_schemas:
            return None
        filtered = [t for t in self.tool_schemas if t.get("name") in allowed]
        return filtered if filtered else None

    


    _NO_TOOL_CALL_GUARD_PROMPT = (
        "[NO_TOOL_GUARD] You have produced text-only responses for multiple consecutive turns. "
        "You MUST call a tool now. If you cannot make further progress, call "
        "phase_done_and_summarize to preserve your work and continue in the next phase."
    )

    def _get_phase_done_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Return tool schemas restricted to phase_done_and_summarize only."""
        return self._expose_only_tool_names({"phase_done_and_summarize"})

    def _inject_no_tool_guard_if_needed(self) -> None:
        """Inject guard prompt for no-tool-call streak."""
        self._inject_user_nudge_unless_recent_contains(
            "[NO_TOOL_GUARD]", self._NO_TOOL_CALL_GUARD_PROMPT
        )

    def _inject_user_nudge_unless_recent_contains(self, marker_substring: str, message: str) -> None:
        """Append *message* as a user nudge unless a recent user turn already contains *marker_substring*."""
        if any(
            isinstance(m, dict) and m.get("role") == "user"
            and marker_substring in str(m.get("content", ""))
            for m in self.conversation_history[-4:]
        ):
            return
        safe_add_user_prompt(self.conversation_history, message)

    def _completion_tail_tool_names(self) -> Set[str]:
        """Return the tool-name set for the global real-iteration tail (policy), not filtered schemas.

        Used together with `_expose_only_tool_names` (generic filter). Keeping both avoids conflating
        which names this policy allows with how to subset `self.tool_schemas` for an arbitrary name set.
        """
        if any(t.get("name") == "finalize_optimization" for t in (self.tool_schemas or [])):
            names: Set[str] = {"finalize_optimization"}
        else:
            names: Set[str] = {"task_done"}
        return names

    def _get_completion_only_tools_for_real_budget(self) -> Optional[List[Dict[str, Any]]]:
        """Tools allowed in the final global-iteration window (task end + optimizer finalize)."""
        return self._expose_only_tool_names(self._completion_tail_tool_names())

    # =====================================================================
    # Loop-time user nudges (dedup guards, global budget, no-tool-call, phase warnings)
    # =====================================================================

    def _inject_task_done_only_guard_if_needed(self) -> None:
        """Inject a strong nudge when only completion tools are exposed (real-iteration tail)."""
        completion = self._get_completion_only_tools_for_real_budget() or []
        cnames = {t.get("name") for t in completion if t.get("name")}
        if "finalize_optimization" in cnames:
            msg = (
                "[REAL_BUDGET_GUARD_TASK_DONE] Only `finalize_optimization` is available. "
                "It runs mandatory configuration and plan checks; deliver a validated step before the budget ends."
            )
        else:
            msg = (
                "[REAL_BUDGET_GUARD_TASK_DONE] Only `task_done` is available. "
                "Call it now with your best final answer (or best-effort summary if incomplete)."
            )
        self._inject_user_nudge_unless_recent_contains('[REAL_BUDGET_GUARD_TASK_DONE]', msg)

    def _add_max_real_iteration_budget_notice(self, real_left: int) -> None:
        """Start-of-turn notice when the global real-iteration budget is almost exhausted."""
        if self._max_real_warn_threshold <= 0 or real_left <= 0:
            return
        self._safe_add_user_prompt_from_agent_template(
            "real_iteration_warning_prompt.md",
            {
                "real_left": str(real_left),
                "max_real_iterations": str(self.max_real_iterations),
            },
        )


    def _add_no_tool_call_message(self) -> None:
        """Inject an escalating nudge when the LLM produces text without tool calls.

        Escalation levels based on ``_consecutive_no_tool_calls``:
          1  — gentle reminder
          2  — direct instruction with examples
          3+ — urgent warning that forced phase_done is imminent
        Subclasses may override for task-specific wording.
        """
        count = getattr(self, "_consecutive_no_tool_calls", 1)
        limit = self.max_consecutive_no_tool_calls

        if count <= 1:
            msg = "You must call a tool to make progress. Choose the most relevant tool and proceed."
        elif count < limit:
            msg = (
                f"No tool call detected for {count} consecutive turns. "
                "You MUST call a tool now — for example: read_file, search, "
                "or phase_done_and_summarize if the current phase is complete."
            )
        else:
            msg = (
                f"WARNING: {count}/{limit} consecutive text-only turns reached. "
                "Forced phase_done will trigger on the next turn. "
                "Call a tool NOW or call phase_done_and_summarize to preserve your progress."
            )
        safe_add_user_prompt(self.conversation_history, msg)
    
    def _add_max_iteration_warning(self, rounds_left: int) -> None:
        """
        Add a warning prompt when iterations are nearly exhausted (subclasses may override).
        """
        self._safe_add_user_prompt_from_agent_template(
            "warning_prompt.md",
            {"rounds_left": str(rounds_left)},
        )

    # =====================================================================
    # LLM output parsing (configurable response tags)
    # =====================================================================

    def _parse_response_tags(self, text: str) -> Dict[str, str]:
        """Extract <tag>content</tag> pairs from LLM text output.

        Configured via config['response_tags'] = ['reflection', 'plan', ...].
        Returns empty dict if no tags configured or none found.
        """
        if not self._response_tags or not text:
            return {}
        extracted: Dict[str, str] = {}
        for tag in self._response_tags:
            match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
            if match:
                extracted[tag] = match.group(1).strip()
        return extracted

    # =====================================================================
    # Tool schemas, argument validation, and batch execution
    # =====================================================================
    async def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Build the full list of tool schemas (MCP tools + internal tools).

        Assumes the MCP client is already connected.
        """
        schemas: List[Dict[str, Any]] = []
        
        # 1. Get all tools from the MCP client
        try:
            mcp_tool_schemas = await self.mcp_client.list_tools()
        except Exception as e:
            await self.message_emitter.emit(MessageType.ERROR, f"Failed to load MCP tools: {str(e)}")
            mcp_tool_schemas = []
        
        # 1.1 Record all MCP tool names
        self.mcp_tool_names = {schema.get("name") for schema in mcp_tool_schemas if schema.get("name")}

        # 2. Get internal tool schemas
        internal_schemas = collect_tool_schemas(self)
        internal_tool_names: Set[str] = {schema.get("name") for schema in internal_schemas if schema.get("name")}

        # 3. Validate available_tools config
        if self.available_tool_names:
            all_tool_names: Set[str] = self.mcp_tool_names | internal_tool_names
            unknown_tools = self.available_tool_names - all_tool_names
            if unknown_tools:
                raise ValueError(
                    f"Configured 'available_tools' contain tools not found in MCP or internal schemas: "
                    f"{sorted(unknown_tools)}"
                )

        # 4. Filter tool schemas to only expose tools listed in available_tools

        # 4.1 MCP tools
        for schema in mcp_tool_schemas:
            tool_name = schema.get("name")
            if not tool_name:
                continue
            if tool_name not in self.available_tool_names:
                continue
            schemas.append(schema)

        # 4.2 Internal tools
        for schema in internal_schemas:
            tool_name = schema.get("name")
            if not tool_name:
                continue
            if tool_name not in self.available_tool_names:
                continue
            # Only expose assign_task_to_agent when allowed_subagents is configured
            if tool_name == "assign_task_to_agent" and not self.allowed_subagents:
                continue
            schemas.append(schema)

        return schemas

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Validate LLM-provided tool arguments against the tool's inputSchema.

        Also fills in schema-declared defaults for any missing optional parameters
        so the tool implementation always receives a complete argument dict.

        Returns:
            None if validation passes; otherwise an error description string.
        """
        if not self.tool_schemas:
            return None
        schema = next((s for s in self.tool_schemas if s.get("name") == tool_name), None)
        if not schema:
            return None
        input_schema = schema.get("inputSchema") or {}
        properties = input_schema.get("properties") or {}
        required = input_schema.get("required") or []
        if not isinstance(arguments, dict):
            return (
                f"Tool '{tool_name}' expects a JSON object as arguments. "
                f"Received: {type(arguments).__name__}. Please pass a valid object with the required keys."
            )

        # Fill in schema-declared defaults for missing optional parameters
        for param_name, param_schema in properties.items():
            if param_name not in arguments and "default" in param_schema:
                arguments[param_name] = param_schema["default"]

        args_keys = set(arguments.keys())
        prop_keys = set(properties.keys())
        missing = [k for k in required if k not in args_keys or arguments.get(k) is None]
        if missing:
            return (
                f"Tool '{tool_name}' is missing required argument(s): {missing}. "
                f"Required: {required}. Please call this tool again with all required arguments."
            )
        if input_schema.get("additionalProperties") is False and args_keys - prop_keys:
            extra = list(args_keys - prop_keys)
            return (
                f"Tool '{tool_name}' does not accept argument(s): {extra}. "
                f"Allowed: {list(prop_keys)}. Please remove the extra arguments and retry."
            )
        return None

    async def execute_tool_calls_batch(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a batch of tool calls.

        Steps: (1) execute all calls, (2) append results to conversation history,
        (3) run all post-execution hooks.
        """

        def _secs(t0: float) -> float:
            return round(time.perf_counter() - t0, 6)

        # ── Phase 0: parse all tool calls and emit invocation events ──
        parsed_calls: List[Tuple[str, Optional[str], Dict[str, Any], Dict[str, Any]]] = []
        for tool_call in tool_calls:
            tool_call = tool_call if isinstance(tool_call, dict) else {}
            func = tool_call.get("function") or {}
            tool_call_id = tool_call.get("id")
            tool_name = func.get("name") if isinstance(func, dict) else None
            arguments_str = func.get("arguments", "{}") if isinstance(func, dict) else "{}"
            t_parse = time.perf_counter()
            try:
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else (
                    arguments_str if isinstance(arguments_str, dict) else {}
                )
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            parse_seconds = _secs(t_parse)
            parsed_calls.append((tool_call_id, tool_name, arguments, {"parse_arguments_seconds": parse_seconds}))

        # Emit all tool invocation events (sequential to preserve log order)
        for tool_call_id, tool_name, arguments, phase_timings in parsed_calls:
            t_emit = time.perf_counter()
            await self.message_emitter.emit_tool(
                tool_name=tool_name or "(invalid)",
                tool_call_id=tool_call_id,
                **({"arguments": arguments} if tool_name else {"success": False, "data": None, "error": "Tool call missing 'name'"}),
            )
            phase_timings["emit_tool_call_event_seconds"] = _secs(t_emit)

        # ── Phase 1: execute all tools concurrently ──
        async def _execute_one(
            tool_call_id: str,
            tool_name: Optional[str],
            arguments: Dict[str, Any],
            phase_timings: Dict[str, Any],
        ) -> Tuple[str, Optional[str], Dict[str, Any], Any, float, Dict[str, Any]]:
            if not tool_name:
                return (tool_call_id, tool_name, {}, {"success": False, "data": None, "error": "Tool call missing 'name' or invalid structure."}, 0.0, phase_timings)
            if tool_name not in self.available_tool_names:
                return (tool_call_id, tool_name, arguments, {"success": False, "data": None, "error": f"Tool '{tool_name}' is not allowed for this agent."}, 0.0, phase_timings)

            _tool_t0 = time.perf_counter()
            mcp_detail: Dict[str, float] = {}
            if tool_name in self.mcp_tool_names:
                result = await self.mcp_client.call_tool(
                    tool_name, arguments, timing_detail=mcp_detail
                )
                if mcp_detail:
                    phase_timings["mcp_client_phases_seconds"] = mcp_detail
            else:
                t_val = time.perf_counter()
                validation_error = self._validate_tool_arguments(tool_name, arguments)
                phase_timings["internal_validate_arguments_seconds"] = _secs(t_val)
                if validation_error:
                    result = {"success": False, "data": None, "error": validation_error}
                    phase_timings["internal_invoke_seconds"] = 0.0
                else:
                    tool_method = getattr(self, tool_name)
                    is_async = inspect.iscoroutinefunction(tool_method)
                    if not is_async and hasattr(tool_method, "__wrapped__"):
                        is_async = inspect.iscoroutinefunction(tool_method.__wrapped__)
                    t_inv = time.perf_counter()
                    try:
                        if is_async:
                            result = await tool_method(**arguments)
                        else:
                            result = tool_method(**arguments)
                    except Exception as e:
                        result = {"success": False, "data": None, "error": f"Tool '{tool_name}' raised: {str(e)}"}
                    phase_timings["internal_invoke_seconds"] = _secs(t_inv)
            tool_duration = time.perf_counter() - _tool_t0
            phase_timings["execute_core_seconds"] = round(tool_duration, 6)
            return (tool_call_id, tool_name, arguments, result, tool_duration, phase_timings)

        if len(parsed_calls) == 1:
            raw_results = [await _execute_one(*parsed_calls[0])]
        elif any(pc[1] in {"task_done", "finalize_optimization", "phase_done_and_summarize"} for pc in parsed_calls):
            # Finish edits and observations before validating or replacing context.
            ordered = sorted(enumerate(parsed_calls), key=lambda item: item[1][1] in {"task_done", "finalize_optimization", "phase_done_and_summarize"})
            raw_results = [None] * len(parsed_calls)
            for index, call in ordered:
                raw_results[index] = await _execute_one(*call)
        else:
            raw_results = list(await asyncio.gather(
                *(_execute_one(*pc) for pc in parsed_calls),
                return_exceptions=True,
            ))

        # ── Phase 2: emit results and collect hooks (sequential for ordering) ──
        results: List[Tuple[Any, ...]] = []
        post_hooks = []
        for i, raw in enumerate(raw_results):
            if isinstance(raw, Exception):
                tc_id = parsed_calls[i][0]
                tc_name = parsed_calls[i][1]
                raw = (tc_id, tc_name, {}, {"success": False, "data": None, "error": f"Tool execution exception: {raw}"}, 0.0, {})

            tool_call_id, tool_name, arguments, result, tool_duration_seconds, phase_timings = raw
            if isinstance(result, dict):
                success = result.get("success", False)
                data = result.get("data")
                error = result.get("error")
            else:
                success = True
                data = result
                error = None

            t_emit_res = time.perf_counter()
            await self.message_emitter.emit_tool(
                tool_name=tool_name or "(invalid)",
                tool_call_id=tool_call_id,
                success=success,
                data=data,
                error=error,
            )
            phase_timings["emit_tool_result_event_seconds"] = _secs(t_emit_res)
            results.append((tool_call_id, tool_name, arguments, result, tool_duration_seconds, phase_timings))

            if isinstance(result, dict) and "post_tool_call_hook" in result:
                post_hooks.append(result["post_tool_call_hook"])
        
        # 2. Append all tool results to conversation history (including per-tool duration for trajectory JSONL)
        for tool_call_id, tool_name, arguments, result, tool_duration_seconds, phase_timings in results:
            t_fmt = time.perf_counter()
            # Format tool result content and determine success status
            if isinstance(result, dict):
                success = result.get("success", False)
                if success:
                    content = str(result.get("data", ""))
                else:
                    error_msg = result.get("error", "Unknown error")
                    content = f"Error: {error_msg}"
                    data = result.get("data")
                    if isinstance(data, dict) and data:
                        hint = data.get("hint_message") or data.get("hint")
                        if hint:
                            content += f"\nHint: {hint}"
                        stdout = data.get("stdout", "")
                        stderr = data.get("stderr", "")
                        if stdout:
                            content += f"\n\nStdout:\n{stdout}"
                        if stderr:
                            content += f"\n\nStderr:\n{stderr}"
            else:
                success = True
                content = str(result)

            # Truncate overly long tool results with tool-specific guidance
            max_tool_result_chars = int(self.config.get("max_tool_result_chars_for_context", 20000))
            if len(content) > max_tool_result_chars:
                truncated_content = content[:max_tool_result_chars]
                guidance = _TRUNCATION_GUIDANCE.get(
                    tool_name,
                    "Retry with more selective parameters to read only the necessary subset.",
                )
                content = (
                    f"{truncated_content}\n\n"
                    f"OUTPUT TRUNCATED ({len(content)} chars, showing first {max_tool_result_chars}). "
                    f"{guidance}"
                )
            format_truncate_content_seconds = _secs(t_fmt)

            # Add to conversation history (OpenAI tool message format)
            tool_message = {
                "role": "tool",
                "content": content,
            }
            if tool_call_id:
                tool_message["tool_call_id"] = tool_call_id

            # Dev / JSONL only: ``_tool_metadata`` is stripped before LLM calls (underscore rule).
            wall_first_loop = (
                float(phase_timings.get("parse_arguments_seconds", 0.0))
                + float(phase_timings.get("emit_tool_call_event_seconds", 0.0))
                + float(tool_duration_seconds)
                + float(phase_timings.get("emit_tool_result_event_seconds", 0.0))
            )
            tool_message["_tool_metadata"] = {
                "success": success,
                "duration_seconds": float(tool_duration_seconds),
                "phase_timings_seconds": {
                    **phase_timings,
                    "format_truncate_content_seconds": format_truncate_content_seconds,
                    "wall_first_loop_emit_parse_execute_emit_seconds": round(wall_first_loop, 6),
                    "wall_including_format_seconds": round(
                        wall_first_loop + float(format_truncate_content_seconds),
                        6,
                    ),
                },
            }

            stamp_trajectory_timestamp(tool_message)
            self.conversation_history.append(tool_message)
        
        # Preserve the completed turn before a hook resets or compresses history.
        if post_hooks:
            await self._persist_pending_history()

        # 3. Execute all post-hooks
        for hook in post_hooks:
            fn = hook.func if isinstance(hook, partial) else hook
            if inspect.iscoroutinefunction(fn):
                await hook()
            else:
                hook()
        
        # Return results (strip hook field to prevent external access)
        return_results = []
        for _, _, _, result, _, _ in results:
            if isinstance(result, dict):
                result_copy = result.copy()
                result_copy.pop("post_tool_call_hook", None)
                return_results.append(result_copy)
            else:
                return_results.append(result)
        
        return return_results

    # =====================================================================
    # Prompt files & conversation seed (system + initial user)
    # =====================================================================

    def _load_agent_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "agent" / filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _safe_add_user_prompt_from_agent_template(self, filename: str, placeholders: Dict[str, str]) -> None:
        """Load ``prompts/agent/<filename>``, replace ``{key}`` placeholders, append as user message."""
        text = self._load_agent_prompt(filename)
        for key, val in placeholders.items():
            text = text.replace("{" + key + "}", val)
        safe_add_user_prompt(self.conversation_history, text)

    async def _build_system_prompt(self) -> None:
        """
        Build and add the system prompt to conversation history (subclasses may override).
        Loads from template; ``system_skills`` inject protocol bodies; ``optional_skills``
        inject name + description only.
        """
        # Extract available tool names for expanding {{tool}} placeholders in protocol bodies
        available_tool_names = None
        if self.tool_schemas:
            available_tool_names = {schema.get("name") for schema in self.tool_schemas if schema.get("name")}
        
        system_prompt = build_system_prompt_with_skills(
            prompt_template_path=self.prompt_dict["system"],
            skill_dir=self.skill_dir,
            system_skills=self.system_skills,
            optional_skills=self.optional_skills,
            available_tool_names=available_tool_names,
            global_skill_dir=self.global_skill_dir
        )
        
        # Read todo file and append (path configurable via config["todo_file_path"])
        agent_name = self.message_emitter.context.get("agent_name")
        todo_file_path = self.config.get("todo_file_path") or f'draft/{agent_name}_todo.md'
        # Read the todo file via read_file tool (without line numbers)
        todo_result = await self.mcp_client.call_tool(
            "read_file",
            {
                "file_path": todo_file_path,
                "include_line_numbers": False
            }
        )
        todo_content = todo_result.get("data") if todo_result.get("success") else None
        if todo_content is None:
            todo_content = f"(unavailable: {todo_result.get('error', 'unknown')})" if todo_result.get("error") else "(empty or unavailable)"
        system_prompt += f"\n\n---\n\n**Existing TODOs from file ({todo_file_path})**:\n{todo_content}"

        # List file tree and append
        file_tree_result = await self.mcp_client.call_tool(
            "list_file_tree",
            {
                "max_depth": 2
            }
        )
        file_tree_content = file_tree_result.get("data") if file_tree_result.get("success") else None
        if file_tree_content is None:
            file_tree_content = f"(unavailable: {file_tree_result.get('error', 'unknown')})" if file_tree_result.get("error") else "(unavailable)"
        system_prompt += f"\n\n---\n\n**Current workspace file tree (max-depth=2 via `list_file_tree` tool)**:\n{file_tree_content}"

        system_msg = {"role": "system", "content": system_prompt}
        stamp_trajectory_timestamp(system_msg)
        self.conversation_history.append(system_msg)

    async def _build_initial_message_from_task_input(self, task: str) -> None:
        """
        Build and add the initial user message from the task (subclasses may override).

        If ``preload_skills`` is configured, their full SKILL.md bodies are
        appended to the initial user message so they are present from the
        first turn without bloating the system prompt.

        Robustness: if the loaded template does NOT contain the ``${task}``
        placeholder, the raw *task* text is **prepended** so that the
        sub-agent always receives the parent's task_content — even when the
        optimizer writes a malformed initial_message.md.
        """
        prompt_dict = self.prompt_dict or {}
        custom_initial_message_path = prompt_dict.get("initial_message")
        
        if custom_initial_message_path and Path(custom_initial_message_path).exists():
            with open(custom_initial_message_path, "r", encoding="utf-8") as f:
                template = f.read().strip()
        else:
            template = self._load_agent_prompt("default_initial_message.md")

        if "${task}" in template:
            initial_message = template.replace("${task}", task)
        else:
            logger.warning(
                "[%s] initial_message template at '%s' does NOT contain ${task} placeholder. "
                "Auto-prepending the task content to ensure the agent receives it.",
                self._agent_name,
                custom_initial_message_path or "<default>",
            )
            initial_message = (
                f"## Task\n\n{task}\n\n---\n\n"
                f"## Additional Instructions\n\n{template}"
            )

        if self.preload_skills and self.skill_dir:
            from .utils.prompt_utils import load_skill
            blocks: List[str] = []
            for skill_name in self.preload_skills:
                try:
                    skill_dict = load_skill(skill_name, self.skill_dir, self.global_skill_dir)
                    body = skill_dict.get("content") or ""
                    if body.strip():
                        blocks.append(
                            f"### Preloaded protocol: {skill_dict.get('name', skill_name)}\n\n{body}"
                        )
                except FileNotFoundError:
                    logger.warning("Preload skill %r not found; skipped.", skill_name)
            if blocks:
                initial_message += "\n\n---\n\n" + "\n\n---\n\n".join(blocks)

        user_msg = {"role": "user", "content": initial_message}
        stamp_trajectory_timestamp(user_msg)
        self.conversation_history.append(user_msg)

    # =====================================================================
    # Per-LLM-step tool list (phase-local cap; execute_task applies further overrides)
    # =====================================================================
    def _get_tools_for_iteration(self, iteration: int) -> Optional[List[Dict[str, Any]]]:
        """Return tool schemas to expose to LLM for the current iteration."""
        if not self.tool_schemas:
            return None

        if iteration <= self.max_iterations - 2:
            return self.tool_schemas

        # Near end of phase: only allow phase_done or task_done so the agent wraps up
        return self._expose_only_tool_names({"phase_done_and_summarize", "task_done"})

    # =====================================================================
    # ReAct entry: main task loop
    # =====================================================================
    async def _persist_pending_history(self) -> None:
        """Append unseen messages to the full trajectory before any context reset."""
        if self._last_saved_history_length > len(self.conversation_history):
            self._last_saved_history_length = 0
        messages = self.conversation_history[self._last_saved_history_length:]
        if messages and await self.message_emitter.emit(
            message_type=MessageType.CONVERSATION_HISTORY,
            metadata={"iteration": self.iteration, "real_iteration": self.real_iteration, "history": messages},
        ):
            self._last_saved_history_length = len(self.conversation_history)

    async def execute_task(
        self,
        task: str,
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a task via the ReAct loop.

        Args:
            task: The task description / follow-up instruction.
            resume: If True, keep existing conversation_history and append a
                follow-up user message instead of resetting. The agent retains
                its prior memory and tools; only loop counters are reset.

        Returns:
            {"success": bool, "data": Any, "error": Optional[str]}

        Optional config (proactive LLM context compression, parallel to other projects'
        ``token_threshold`` + ``compress``):
            - ``llm_context_compression_token_threshold`` (int): when > 0 and estimated
              tokens (chars/3 heuristic) reach this value, an extra LLM pass may summarize
              middle history via ``compress_conversation_with_llm`` before the main call.
            - ``llm_context_compression`` (bool): set ``False`` to disable even if threshold set.
            - ``llm_context_compression_min_shrink_ratio`` (float, default 0.9): compression is
              applied only if the new heuristic token estimate is below ``ratio * est_before``
              (avoids no-op swaps that would break traj/context for no benefit).
            Context compression uses the **same** LLM as the agent (``self.llm`` / ``llm_config``).
        """
        self._current_task = task
        await self.message_emitter.emit_task(status="start", task=task)

        if self.tool_schemas is None:
            self.tool_schemas = await self._build_tool_schemas()

        if resume and self.conversation_history:
            # Resume mode: sync save pointer to current length so the incremental
            # persist only captures the new follow-up + subsequent messages.
            self._last_saved_history_length = len(self.conversation_history)
            follow_up = (
                f"[Follow-up task — you retain full context from the previous task.]\n\n"
                f"{task}"
            )
            safe_add_user_prompt(self.conversation_history, follow_up)
        else:
            # Fresh start: reset conversation and build from scratch
            self.conversation_history = []
            self._last_saved_history_length = 0

            await self.message_emitter.emit(
                message_type=MessageType.TOOL_SCHEMA,
                metadata={"tool_schemas": self.tool_schemas}
            )

            await self._build_system_prompt()
            await self._build_initial_message_from_task_input(task)

        # ReAct loop state (always reset per invocation)
        self.iteration = 0
        self.real_iteration = 0
        self.is_task_complete = False
        self.task_done_summary = None
        self._consecutive_llm_failures = 0
        self._consecutive_no_tool_calls = 0
        self._fatal_error = None

        while self.iteration < self.max_iterations and self.real_iteration < self.max_real_iterations and not self.is_task_complete:
            self.iteration += 1
            self.real_iteration += 1
            self.message_emitter.update_iteration(self.iteration, self.real_iteration)

            real_left = self.max_real_iterations - self.real_iteration
            if self._max_real_warn_threshold > 0 and 0 < real_left <= self._max_real_warn_threshold:
                self._add_max_real_iteration_budget_notice(real_left)

            # ── Tool-selection & context-safety pipeline ─────────────────
            # Three-tier context management (soft < hard < must < LLM max):
            #
            #   S1. Baseline tools (phase-local)
            #   S2. MUST limit (ctx > must): force LLM summary compression
            #   S3. HARD limit (ctx > hard): restrict tools to {phase_done, task_done}
            #   S4. SOFT limit (ctx > soft): warning nudge only, no tool change
            #   S5. No-tool-call streak guard (may force phase_done)
            #   S6. Global real-iteration tail (may force task_done)
            #
            # Ordering rationale:
            #   - S2 compresses first → S3/S4 evaluate the post-compression size.
            #   - S3 restricts tools only if context is still over hard after S2.
            #   - S4 warning is harmless (no tool change) and runs on accurate size.
            #   - S6 (tail guard) runs last as the highest-priority override.

            # ── S1. Baseline tools ────────────────────────────────────────
            tools_for_this_iteration = self._get_tools_for_iteration(self.iteration)
            await self._persist_pending_history()

            # ── S2. MUST limit: force LLM summary compression ─────────────
            if await self._force_compress_context():
                await self.message_emitter.emit(
                    MessageType.WARNING,
                    f"Context must limit exceeded ({self._context_must_limit} chars); "
                    "conversation compressed via LLM summary."
                )

            # ── S3. HARD limit: restrict to phase_done + task_done ────────
            if self._check_context_hard_limit():
                hard_tools = self._get_hard_limit_tools()
                if hard_tools:
                    self._inject_hard_limit_guard_if_needed()
                    tools_for_this_iteration = hard_tools
                await self.message_emitter.emit(
                    MessageType.WARNING,
                    f"Context hard limit exceeded "
                    f"({self._estimate_conversation_history_size()}/{self._context_hard_limit} chars); "
                    "tools restricted to phase_done_and_summarize + task_done."
                )

            # ── S4. SOFT limit: warning nudge only ────────────────────────
            if self._check_context_soft_warning():
                await self.message_emitter.emit(
                    MessageType.WARNING,
                    f"Context approaching soft limit "
                    f"({self._estimate_conversation_history_size()}/{self._context_soft_limit} chars); "
                    "agent warned."
                )

            # ── S5. No-tool-call streak: force phase_done ─────────────────
            reasons: List[str] = []
            if self._should_force_phase_done(reasons):
                phase_tools = self._get_phase_done_tools()
                if phase_tools:
                    self._inject_no_tool_guard_if_needed()
                    tools_for_this_iteration = phase_tools
                await self.message_emitter.emit(
                    MessageType.ERROR,
                    f"Force phase_done triggered: {'; '.join(reasons)}"
                )

            # ── S6. Global real-iteration tail: completion tools only ─────
            if self._max_real_task_done_only_threshold > 0 and real_left <= self._max_real_task_done_only_threshold:
                completion_only = self._get_completion_only_tools_for_real_budget()
                if completion_only:
                    tools_for_this_iteration = completion_only
                    self._inject_task_done_only_guard_if_needed()

            # ── LLM call ─────────────────────────────────────────────────
            _llm_wall_start = time.perf_counter()
            try:
                response = await self.llm.generate(
                    messages=self.conversation_history,
                    tools=tools_for_this_iteration
                )

            except ContextOverflowError as e:
                # LLM provider rejected the request — context too long.
                # Compress (LLM summary → tail-drop fallback) and retry.
                self._consecutive_llm_failures = 0
                await self.message_emitter.emit(
                    MessageType.ERROR,
                    f"LLM context overflow: {e}"
                )

                compressed = await self._force_compress_context(skip_threshold=True)
                if compressed:
                    if self._check_context_hard_limit():
                        hard_tools = self._get_hard_limit_tools()
                        if hard_tools:
                            self._inject_hard_limit_guard_if_needed()
                            tools_for_this_iteration = hard_tools
                    continue

                # Nothing could be compacted → abort
                self._fatal_error = f"Context overflow and cannot compact further: {e}"
                break

            except LLMError as e:
                err_text = str(e)

                error_template = self._load_agent_prompt("llm_error_prompt.md")
                safe_add_user_prompt(self.conversation_history, error_template.replace("{error}", err_text))
                await self.message_emitter.emit(MessageType.ERROR, f"LLM generation failed: {err_text}")

                self._consecutive_llm_failures += 1
                if self._consecutive_llm_failures >= self.max_consecutive_llm_failures:
                    self._fatal_error = (
                        f"Agent aborted after {self._consecutive_llm_failures} consecutive LLM failures. "
                        f"Last error: {err_text}"
                    )
                    await self.message_emitter.emit(MessageType.ERROR, self._fatal_error)
                    break
                continue

            _llm_wall_secs = time.perf_counter() - _llm_wall_start

            # ── Response validation ──────────────────────────────────────
            if response is None or not isinstance(response, dict):
                self._consecutive_llm_failures += 1
                invalid_prompt = self._load_agent_prompt("invalid_response_prompt.md")
                safe_add_user_prompt(self.conversation_history, invalid_prompt)
                if self._consecutive_llm_failures >= self.max_consecutive_llm_failures:
                    self._fatal_error = f"Agent aborted: {self._consecutive_llm_failures} consecutive invalid LLM responses."
                    break
                continue

            self._consecutive_llm_failures = 0

            text_content = response.get("text_content", "")
            raw_tool_calls = response.get("tool_calls") or []
            tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []

            # ── Build assistant message ──────────────────────────────────
            assistant_message: Dict[str, Any] = {"role": "assistant", "content": text_content}

            if response.get("metadata"):
                assistant_message["_llm_metadata"] = response["metadata"]

            # Log-only timing for trajectory JSONL: underscore keys are never sent to the
            # model (see optimas_client/src/llm/base.py::_clean_messages_for_api).
            _meta = response.get("metadata")
            if isinstance(_meta, dict) and _meta.get("duration") is not None:
                assistant_message["_llm_duration_seconds"] = float(_meta["duration"])
            else:
                assistant_message["_llm_duration_seconds"] = float(_llm_wall_secs)

            # Extensible tag extraction
            parsed_tags = self._parse_response_tags(text_content)
            if parsed_tags:
                assistant_message["_parsed_tags"] = parsed_tags

            if tool_calls:
                openai_tool_calls = []
                for tc in tool_calls:
                    tc = tc if isinstance(tc, dict) else {}
                    tc_func = tc.get("function") or {}
                    openai_tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc_func.get("name", ""),
                            "arguments": tc_func.get("arguments", "{}")
                        }
                    })
                    for key in ("extra_content", "thought_signature"):
                        if key in tc:
                            openai_tool_calls[-1][key] = tc[key]
                assistant_message["tool_calls"] = openai_tool_calls
            stamp_trajectory_timestamp(assistant_message)
            self.conversation_history.append(assistant_message)

            # ── Dispatch: tool calls vs text-only ────────────────────────
            if not tool_calls:
                display_content = text_content.strip() if text_content else "(no text content)"
                await self.message_emitter.emit(
                    message_type=MessageType.LLM_RESPONSE,
                    content=display_content,
                    metadata={"text_content": text_content, "tool_calls_count": 0}
                )
                self._consecutive_no_tool_calls += 1
                self._add_no_tool_call_message()
            else:
                self._consecutive_no_tool_calls = 0
                if text_content and text_content.strip():
                    await self.message_emitter.emit(
                        message_type=MessageType.LLM_RESPONSE,
                        content=text_content.strip(),
                        metadata={"text_content": text_content, "tool_calls_count": len(tool_calls)}
                    )
                await self.execute_tool_calls_batch(tool_calls)

            # ── End-of-phase iteration warning ───────────────────────────
            rounds_left = self.max_iterations - self.iteration
            if 0 < rounds_left <= 3:
                self._add_max_iteration_warning(rounds_left)

            # ── Persist new conversation messages ────────────────────────
            # Guard: compaction or phase_done may shrink history below the pointer
            if self._last_saved_history_length > len(self.conversation_history):
                self._last_saved_history_length = 0
            new_messages = self.conversation_history[self._last_saved_history_length:]
            if new_messages:
                ok = await self.message_emitter.emit(
                    message_type=MessageType.CONVERSATION_HISTORY,
                    metadata={
                        "iteration": self.iteration,
                        "real_iteration": self.real_iteration,
                        "history": new_messages
                    }
                )
                if ok:
                    self._last_saved_history_length = len(self.conversation_history)

        await self._persist_pending_history()
        # ── Post-loop exit handling ──────────────────────────────────────
        if self.is_task_complete:
            assert self.task_done_summary is not None
            await self.message_emitter.emit_task(status="done", task=task, summary=self.task_done_summary)
            return {"success": True, "data": self.task_done_summary, "error": None}

        if self._fatal_error:
            await self.message_emitter.emit_task(status="failed", task=task, error=self._fatal_error)
            return {"success": False, "data": None, "error": self._fatal_error}

        max_iter_error = f"Task still not finished after maximum iterations ({self.max_iterations})"
        await self.message_emitter.emit_task(status="failed", task=task, error=max_iter_error)
        return {"success": False, "data": None, "error": max_iter_error}
