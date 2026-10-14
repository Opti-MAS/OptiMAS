"""Trajectory file output handler"""
import asyncio
import json
from pathlib import Path
from .base import BaseHandler
from ..types import StructuredMessage, MessageType
from ...utils.exceptions import MessagingError


class TrajFileHandler(BaseHandler):
    """Trajectory file output handler
    
    Supports dynamic paths: {base_path}/{agent_name}_{invocation_id}_{phase_count}.jsonl
    Always uses append mode and JSON format.
    Only saves conversation_history and tool_schema message types.
    
    ``invocation_id`` is a short hex tag generated per ``execute_task`` call,
    ensuring that repeated fresh creations of the same sub-agent produce
    distinct log file sets.
    
    Args:
        base_path: Base path for the storage folder (relative or absolute), defaults to "trajs"
    """
    
    def __init__(self, base_path: str = "trajs"):
        """
        Initialize the file handler.
        
        Args:
            base_path: Base path for the storage folder (relative or absolute), defaults to "trajs"
        """
        self.base_path = Path(base_path)
    
    async def handle(self, message: StructuredMessage):
        """
        Handle a structured message by writing to file (JSON format, append mode, async).
        
        Only saves conversation_history and tool_schema message types; others are ignored.
        
        For tool_schema: saved as a single JSON line.
        For conversation_history: expands the metadata.history array, saving one message per line.
        
        Extracts agent_name, invocation_id, and phase_count from message.context to build
        dynamic file paths.
        
        Args:
            message: Structured message object
        """
        # Only save conversation_history and tool_schema message types
        if message.type not in (MessageType.CONVERSATION_HISTORY, MessageType.TOOL_SCHEMA):
            return
        
        try:
            agent_name = message.context.get("agent_name", "unknown")
            invocation_id = message.context.get("invocation_id", "")
            phase_count = message.context.get("phase_count", 0)
            
            if invocation_id:
                file_path = self.base_path / f"{agent_name}_{invocation_id}_{phase_count}.jsonl"
            else:
                file_path = self.base_path / f"{agent_name}_{phase_count}.jsonl"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Wrap file writes with asyncio.to_thread
            def _write_file():
                with open(file_path, "a", encoding="utf-8") as f:
                    if message.type == MessageType.TOOL_SCHEMA:
                        # Tool schema: save as a single line containing tool_schemas
                        save_data = {
                            "type": "tool_schema",
                            "timestamp": message.timestamp.isoformat(),
                            "data": message.metadata.get("tool_schemas", [])
                        }
                        f.write(json.dumps(save_data, ensure_ascii=False) + "\n")
                    elif message.type == MessageType.CONVERSATION_HISTORY:
                        # Conversation history: expand the history array, one message per line.
                        # Each message should have ``_trajectory_timestamp`` (UTC ISO8601)
                        # stamped by stamp_trajectory_timestamp when entering history;
                        # underscore-prefixed keys are stripped before sending to LLM.
                        history = message.metadata.get("history", [])
                        
                        for msg in history:
                            # If tool_calls exist, parse arguments from JSON string to object
                            save_msg = msg
                            if "tool_calls" in msg:
                                save_msg = json.loads(json.dumps(msg))  # deep copy to avoid mutating original
                                for tc in save_msg.get("tool_calls", []):
                                    func = tc.get("function", {})
                                    args = func.get("arguments")
                                    if isinstance(args, str):
                                        try:
                                            func["arguments"] = json.loads(args)
                                        except (json.JSONDecodeError, TypeError):
                                            pass  # keep as-is on parse failure
                            
                            # Save directly; ensure_ascii=False preserves non-ASCII characters
                            f.write(json.dumps(save_msg, ensure_ascii=False) + "\n")
            
            await asyncio.to_thread(_write_file)
        except Exception as e:
            raise MessagingError(f"Failed to write to file: {str(e)}") from e
