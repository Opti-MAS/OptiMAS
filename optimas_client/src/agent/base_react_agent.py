"""Worker agent: internal AI tools, sub-agent delegation, and proactive / tail-drop context compression.

The ReAct loop, prompts, and tool batch dispatch live on :class:`~.react_loop_agent.BaseReactAgent`.
"""
import logging
from functools import partial
from typing import Dict, Any, Optional, List

from ..messaging.types import MessageType
from .react_loop_agent import BaseReactAgent
from .utils.tool_decorator import mcp_tool
from .utils.prompt_utils import load_skill as load_skill_object
from .utils.context_management import (
    safe_add_user_prompt,
    format_compressible_messages_for_llm,
)
from .utils.internal_ai_tools import (
    image_qa as _image_qa,
    text_extract as _text_extract,
    text_qa as _text_qa,
)

logger = logging.getLogger(__name__)


class BaseAgent(BaseReactAgent):
    """ReAct agent with internal tools, delegation, and context compression."""

    #---- MCP alike internal tools ----#

    @mcp_tool(
        input_schema={
            "skill_name": {
                "type": "string",
                "description": "Skill id (directory name) or parent/sub_skill path; see system prompt listings.",
            }
        }
    )
    def load_skill(self, skill_name: str) -> Dict[str, Any]:
        """
        Load a configured skill and inject its body after the tool result.
        """
        if not self.skill_dir:
            return {"success": False, "data": None, "error": "No skill directory configured for this agent."}
        allowed = set((self.system_skills or []) + (self.preload_skills or []) + (self.optional_skills or []))
        if skill_name not in allowed:
            return {"success": False, "data": None, "error": "Skill is not configured for this agent."}
        try:
            skill_object = load_skill_object(skill_name, self.skill_dir, self.global_skill_dir)
        except FileNotFoundError as e:
            return {"success": False, "data": None, "error": f"Skill not found: {skill_name}. {e}"}
        except Exception as e:
            return {"success": False, "data": None, "error": f"Failed to load skill '{skill_name}': {e}"}

        def append_skill_to_conversation(agent, skill_object: Dict[str, Any]):
            # called after tool call by post_tool_call_hook
            skill_prompt_template = self._load_agent_prompt("load_skill_prompt.md")
            skill_message = skill_prompt_template.replace("{skill_name}", skill_name).replace("{skill_content}", skill_object.get('content', ''))
            safe_add_user_prompt(agent.conversation_history, skill_message)
        return {
            "success": True,
            "data": f"'{skill_name}' loaded. The user will prompt you the skill usage to continue the task.",
            "post_tool_call_hook": partial(append_skill_to_conversation, self, skill_object)
        }
    
    @mcp_tool(
        input_schema={
            "task_summary": {
                "type": "string",
                "description": "Content returned as the task result. For research/Q&A tasks: give the **final answer** to the user's question and a **brief explanation** (what you found and why). Do NOT use file paths, todo file location, or work-process summary as the main content. For build/code tasks: concise summary of execution results and outcomes."
            }
        }
    )
    def task_done(self, task_summary: str) -> Dict[str, Any]:
        """
        Mark the current task as complete. The task_summary is returned as the task result (e.g. to the benchmark). For Q&A: put the final answer and explanation here, not a file path.
        """
        self.is_task_complete = True
        self.task_done_summary = task_summary
        return {"success": True, "data": "Task marked as complete."}
    
    @mcp_tool(
        input_schema={
            "progress_report": {
                "type": "string",
                "description": "Summary of what was accomplished: key findings, files read/changed, decisions made, and analysis results. Be specific — this text plus an auto-compressed conversation log is what the next phase receives."
            },
            "remaining_tasks": {
                "type": "string",
                "description": "Ordered list of concrete next steps for the next phase. Include guidance such as which files or TODO items to check first."
            }
        }
    )
    def phase_done_and_summarize(self, progress_report: str, remaining_tasks: str) -> Dict[str, Any]:
        """
        Mark phase complete, auto-compress current conversation, and continue with clean new context.
        """
        async def create_new_conversation_history(agent, progress_report: str, remaining_tasks: str) -> None:
            # Auto-compress: extract a detailed summary of the current phase's work
            # before clearing conversation_history
            compressed_summary = await _auto_compress_phase(agent)

            # Build the phase resume message
            summary_section = "\n\n---\n\n**Phase Resume — Continuing from previous phase**\n\n"
            summary_section += f"**Your summary:**\n{progress_report}\n\n"
            summary_section += f"**Remaining tasks:**\n{remaining_tasks}\n"
            if compressed_summary:
                summary_section += (
                    f"\n**Auto-compressed work log from previous phase:**\n"
                    f"{compressed_summary}\n"
                )
            summary_section += (
                "\n---\n\n"
                "**Guidance:** Your TODO file and workspace file tree are in the system prompt above. "
                "Read your TODO file and any working notes you wrote under your workspace "
                "to recall your analysis and decisions. Then continue with the remaining tasks."
            )

            # Re-inject preload_skills so they survive across phase transitions
            preload_skills_section = _build_preload_skills_section(agent)
            if preload_skills_section:
                summary_section += "\n\n---\n\n" + preload_skills_section

            await agent.message_emitter.emit_phase(
                progress_report=progress_report,
                remaining_tasks=remaining_tasks,
                iteration=agent.iteration
            )

            agent.message_emitter.increment_phase()

            # Reset conversation history: rebuild system prompt, inject phase resume
            agent.conversation_history = []
            agent._last_saved_history_length = 0
            await agent._build_system_prompt()
            safe_add_user_prompt(agent.conversation_history, summary_section)

            if agent.tool_schemas:
                await agent.message_emitter.emit(
                    message_type=MessageType.TOOL_SCHEMA,
                    content="",
                    metadata={
                        "tool_schemas": agent.tool_schemas
                    }
                )

            agent.iteration = 0

        return {
            "success": True,
            "data": "Phase summary recorded. You should continue the remaining tasks.",
            "post_tool_call_hook": partial(create_new_conversation_history, self, progress_report, remaining_tasks)
        }

    @mcp_tool(
        input_schema={
            "file_path": {
                "type": "string",
                "description": (
                    "Path to a text file (.txt, .md, .json, .jsonl, .yaml, .py, .csv, .log, etc.) "
                    "or a rich-format document (.pdf, .docx, .pptx, .xlsx, .csv, .html). "
                    "Do NOT pass binary/media files (images, archives, executables)."
                ),
            },
            "question": {
                "type": "string",
                "description": "Question to answer based on the file content."
            }
        }
    )
    async def text_qa(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Answer questions about text or document files using LLM-powered incremental reading.

        Supports plain text files AND rich formats (PDF, DOCX, PPTX, XLSX, CSV, HTML).
        For small files: single-round Q&A.
        For large files: automatically chunks the file, extracts relevant info per chunk
        with rolling context summary, then synthesizes a final answer.
        Output is auto-summarized if it exceeds length limits.
        """
        max_chunk_chars = self.config.get("text_qa_chunk_chars", 16000)
        return await _text_qa(self.mcp_client, self.internal_tools_llm, file_path, question, max_chunk_chars=max_chunk_chars)

    @mcp_tool(
        input_schema={
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Workspace-relative paths to text or document files to analyze in parallel "
                    "(.txt, .md, .json, .jsonl, .yaml, .py, .pdf, .docx, .xlsx, .csv, .html, etc.). "
                    "Not for binary/media formats (images, archives, executables)."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "Task description and/or plan: what to extract from each document "
                    "(time, authority, relevance, key facts as JSON per prompt schema)."
                ),
            },
        }
    )
    async def text_extract(self, file_paths: List[str], task: str) -> Dict[str, Any]:
        """
        Run structured multi-dimensional extraction over multiple text files in parallel.

        Each file is read via MCP (with chunked fallback for oversized reads), split into
        overlapping windows, and processed with bounded concurrent LLM calls. Each chunk
        returns JSON fields (doc_time, source_authority, task_relevance, key_info) when
        the model follows the prompt; parse failures still return ``raw_answer``.
        """
        core = int(self.config.get("text_extract_core_window_chars", 50_000))
        halo = int(self.config.get("text_extract_halo_chars", 200))
        max_chunks = int(self.config.get("text_extract_max_chunks_per_doc", 20))
        conc = int(self.config.get("text_extract_llm_concurrency", 20))
        return await _text_extract(
            self.mcp_client,
            self.internal_tools_llm,
            file_paths,
            task,
            core_window=core,
            halo=halo,
            max_chunks_per_doc=max_chunks,
            max_llm_concurrency=conc,
        )

    @mcp_tool(
        input_schema={
            "file_path": {
                "type": "string",
                "description": (
                    "Path to an image file (.jpg, .jpeg, .png, .gif, .webp) OR an image URL (http/https). "
                    "Do NOT pass PDF, Office documents, or non-image files — use text_qa instead."
                ),
            },
            "question": {
                "type": "string",
                "description": "Question about the image content."
            }
        }
    )
    async def image_qa(self, file_path: str, question: str) -> Dict[str, Any]:
        """
        Answer questions about images using a Vision Language Model (single round).

        Accepts local image paths or remote URLs (auto-downloaded to workspace).
        Rejects non-image files (PDF, documents, etc.) with actionable error messages.
        Output is auto-summarized if excessively long.
        """
        return await _image_qa(self.mcp_client, self.vlm, file_path, question)

    @mcp_tool(
        input_schema={
            "agent_name": {
                "type": "string",
                "description": "Name of the sub-agent to assign the task.",
            },
            "task_content": {
                "type": "string",
                "description": "Task content for the sub-agent to execute."
            },
            "resume_previous": {
                "type": "boolean",
                "description": (
                    "If true, resume the previously stored sub-agent instance (preserving "
                    "its conversation memory). If false (default), create a fresh instance "
                    "and discard any previously stored one."
                ),
                "default": False,
            }
        }
    )
    async def assign_task_to_agent(
        self,
        agent_name: str,
        task_content: str,
        resume_previous: bool = False,
    ) -> Dict[str, Any]:
        """
        Assign a task to a sub-agent for execution.

        Behaviour controlled by *resume_previous*:
          - False (default): create a new sub-agent, discard any stored instance for this name.
          - True: reuse the stored sub-agent (with its full conversation context).
            Falls back to creating new if no stored instance exists.

        Storage invariant: at most **one** instance per (ancestor, subagent_name) pair.
        """
        agent_name = (agent_name or "").strip()
        if not agent_name:
            return {"success": False, "data": None, "error": "agent_name is required and cannot be empty."}
        if agent_name not in self.allowed_subagents:
            allowed = sorted(self.allowed_subagents) if self.allowed_subagents else []
            return {
                "success": False,
                "data": None,
                "error": f"Agent is not allowed to delegate to '{agent_name}'. Allowed subagents for this agent: {allowed}."
            }

        if agent_name in self._ancestor_agents:
            chain = " -> ".join(sorted(self._ancestor_agents)) + f" -> {self._agent_name} -> {agent_name}"
            return {
                "success": False,
                "data": None,
                "error": f"Cycle detected in agent delegation: {chain}. Cannot delegate to an ancestor agent."
            }

        from .factory import create_agent

        stored = self._subagent_store.get(agent_name)

        if resume_previous and stored is not None:
            logger.info(
                "[%s] Resuming stored sub-agent '%s' (history_len=%d)",
                self._agent_name, agent_name, len(stored.conversation_history),
            )
            result = await stored.execute_task(task_content, resume=True)
            # Keep the (potentially updated) instance in the store
            return result

        # Create new — discard any previously stored instance
        if stored is not None:
            logger.info(
                "[%s] Discarding stored sub-agent '%s' and creating new instance.",
                self._agent_name, agent_name,
            )

        sub_agent = create_agent(
            agent_name=agent_name,
            mcp_client=self.mcp_client,
            config=self.global_config,
            ancestor_agents=self._ancestor_agents | {self._agent_name},
        )

        # Store the new instance (replaces any previous one)
        self._subagent_store[agent_name] = sub_agent
        return await sub_agent.execute_task(task_content)


def _build_preload_skills_section(agent) -> Optional[str]:
    """Load preload_skills bodies and return a combined markdown section.

    Called at phase_done_and_summarize to re-inject preloaded protocols into
    the phase resume message, so they survive across phase transitions with
    the same guarantee as system skills.
    """
    preload_skills = getattr(agent, "preload_skills", None)
    skill_dir = getattr(agent, "skill_dir", None)
    if not preload_skills or not skill_dir:
        return None

    from .utils.prompt_utils import load_skill

    blocks: list[str] = []
    for skill_name in preload_skills:
        try:
            skill_dict = load_skill(skill_name, skill_dir, getattr(agent, "global_skill_dir", None))
            body = skill_dict.get("content") or ""
            if body.strip():
                blocks.append(
                    f"### Preloaded protocol: {skill_dict.get('name', skill_name)}\n\n{body}"
                )
        except FileNotFoundError:
            pass
    return "\n\n---\n\n".join(blocks) if blocks else None


async def _auto_compress_phase(agent) -> Optional[str]:
    """Compress the current phase's conversation into a summary string.

    Uses the same LLM compression prompt as the context compressor.
    Returns the summary text, or None if compression fails or there's
    nothing meaningful to compress.
    """
    from .utils.prompt_utils import load_prompt_template
    import os

    history = agent.conversation_history
    if not history or len(history) < 4:
        return None

    working_msgs = []
    found_first_assistant = False
    for msg in history:
        if not found_first_assistant:
            if msg.get("role") == "assistant":
                found_first_assistant = True
                working_msgs.append(msg)
            continue
        working_msgs.append(msg)

    if len(working_msgs) < 2:
        return None

    conversation_text = format_compressible_messages_for_llm(
        working_msgs, tool_body_max=2000,
    )

    compress_prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "prompts", "internal_ai_tools", "context_compression_system_prompt.txt",
    )
    try:
        system_prompt = load_prompt_template(compress_prompt_path)
    except Exception:
        return None

    req = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Compress the following agent conversation ({len(working_msgs)} messages) "
                f"into a detailed summary that preserves all important information:\n\n"
                f"{conversation_text}"
            ),
        },
    ]
    try:
        resp = await agent.llm.generate(messages=req, tools=None)
        summary = (resp.get("text_content") or "").strip()
        return summary if summary else None
    except Exception as exc:
        logger.warning("Phase auto-compress failed: %s", exc)
        return None
