"""Console output handler"""
import asyncio
import sys
import json
from typing import Dict, Any
from .base import BaseHandler
from ..types import StructuredMessage, MessageType


# Global lock to protect console output from interleaving when multiple emitters print concurrently
_console_lock = asyncio.Lock()


class ConsoleHandler(BaseHandler):
    """Console output handler"""
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize the console handler.
        
        Args:
            use_colors: Whether to use colored output
        """
        self.use_colors = use_colors
    
    def _get_color_code(self, message_type: str) -> str:
        """Get color code based on message type."""
        if not self.use_colors:
            return ""
        
        # Map message types to colors
        colors = {
            MessageType.ERROR: "\033[31m",           # red
            MessageType.TASK_FAILED: "\033[31m",     # red
            MessageType.TOOL_RESULT: "\033[33m",     # yellow
            MessageType.TOOL_CALL: "\033[36m",       # cyan
            MessageType.LLM_REQUEST: "\033[34m",     # blue
            MessageType.LLM_RESPONSE: "\033[35m",    # magenta
            MessageType.TASK_START: "\033[32m",       # green
            MessageType.TASK_DONE: "\033[32m",       # green
            MessageType.PHASE_SUMMARY: "\033[36m",   # cyan
            MessageType.ITERATION_START: "\033[90m", # gray
            MessageType.MESSAGE: "\033[37m",         # white
        }
        return colors.get(message_type, "")
    
    def _get_reset_code(self) -> str:
        """Get the ANSI reset code."""
        return "\033[0m" if self.use_colors else ""
    
    def _get_agent_color(self, agent_name: str) -> str:
        """Get the color code for an agent name."""
        if not self.use_colors:
            return ""
        
        # Assign different colors to different agents
        agent_colors = {
            "planner": "\033[35m",   # magenta
            "seeker": "\033[34m",    # blue
            "coder": "\033[32m",     # green
        }
        return agent_colors.get(agent_name.lower(), "\033[37m")  # default white
    
    def _get_type_color(self, message_type: str) -> str:
        """Get the color code for a message type."""
        if not self.use_colors:
            return ""
        
        # Map types to colors
        type_colors = {
            MessageType.TOOL_CALL: "\033[36m",      # cyan
            MessageType.TOOL_RESULT: "\033[33m",    # yellow
            MessageType.MESSAGE: "\033[37m",        # white
        }
        return type_colors.get(message_type, self._get_color_code(message_type))
    
    def _get_type_display(self, message_type: str) -> str:
        """Map MessageType to a user-friendly display name (with emoji)."""
        type_mapping = {
            MessageType.TOOL_CALL: "🔧 tool call",
            MessageType.TOOL_RESULT: "✅ tool response",
            MessageType.MESSAGE: "💬 common message",
            MessageType.ERROR: "❌ error",
            MessageType.TASK_START: "🚀 task start",
            MessageType.TASK_DONE: "✨ task done",
            MessageType.TASK_FAILED: "💥 task failed",
            MessageType.LLM_REQUEST: "🤖 llm request",
            MessageType.LLM_RESPONSE: "💭 llm response",
            MessageType.PHASE_SUMMARY: "📋 phase summary",
            MessageType.ITERATION_START: "🔄 iteration start",
        }
        return type_mapping.get(message_type)
    
    def _format_content(self, message: StructuredMessage) -> str:
        """Format content based on message type."""
        if message.type == MessageType.TOOL_CALL:
            tool_name = message.metadata.get("tool_name", "unknown")
            arguments = message.metadata.get("arguments", {})
            arguments_json = json.dumps(arguments, indent=2, ensure_ascii=False)
            return f"{tool_name}:\n{arguments_json}"
        
        elif message.type == MessageType.TOOL_RESULT:
            data = message.metadata.get("data")
            error = message.metadata.get("error")
            success = message.metadata.get("success", False)
            
            if not success and error:
                parts = [f"Error: {error}"]
                if isinstance(data, dict) and data:
                    stdout = data.get("stdout", "")
                    stderr = data.get("stderr", "")
                    if stdout:
                        parts.append(f"\nStdout:\n{stdout}")
                    if stderr:
                        parts.append(f"\nStderr:\n{stderr}")
                return "".join(parts)
            
            if data is None:
                return ""
            
            if isinstance(data, (dict, list)):
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return str(data)
        
        elif message.type == MessageType.MESSAGE:
            return message.content
        
        # Other types: return raw content
        return message.content
    
    def _get_meta_color_code(self, key: str) -> str:
        """Get color code based on metadata field type."""
        if not self.use_colors:
            return ""
        
        # Map field types to colors
        colors = {
            "tool_name": "\033[36m",      # cyan
            "agent_name": "\033[35m",     # magenta
            "iteration": "\033[33m",      # yellow
            "success": "\033[32m",        # green
            "error": "\033[31m",          # red
            "phase_count": "\033[33m",    # yellow
            "real_iteration": "\033[33m", # yellow
        }
        return colors.get(key, "\033[90m")  # default gray
    
    def _format_message_type(self, message_type: str) -> str:
        """Format message type for display."""
        # Convert underscore-separated to title case
        return message_type.replace("_", " ").title()
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata into a human-readable JSON layout with colors and visual cues.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted string
        """
        if not metadata:
            return ""
        
        # Define field priority and display order
        priority_fields = ["tool_name", "agent_name", "iteration", "phase_count", "real_iteration", "success", "error"]
        
        # Separate priority fields from other fields
        priority_meta = {}
        other_meta = {}
        
        for key, value in metadata.items():
            if key in priority_fields:
                priority_meta[key] = value
            else:
                other_meta[key] = value
        
        # Build formatted output
        lines = []
        reset_code = self._get_reset_code()
        
        # Display priority fields first (with colors)
        if priority_meta:
            for key in priority_fields:
                if key in priority_meta:
                    value = priority_meta[key]
                    color_code = self._get_meta_color_code(key)
                    
                    # Format key name (underscores to spaces, title case)
                    key_display = key.replace("_", " ").title()
                    
                    # Format value
                    if isinstance(value, bool):
                        value_display = "✓" if value else "✗"
                    elif isinstance(value, (dict, list)):
                        value_display = json.dumps(value, ensure_ascii=False, indent=2)
                    else:
                        value_display = str(value)
                    
                    lines.append(f"  {color_code}├─ {key_display}:{reset_code} {value_display}")
        
        # Display other fields in JSON format
        if other_meta:
            # Limit JSON output length to avoid excessive verbosity
            json_str = json.dumps(other_meta, ensure_ascii=False, indent=2)
            if len(json_str) > 500:
                json_str = json_str[:500] + "\n  ... (truncated)"
            
            # Display other fields in gray
            other_color = "\033[90m" if self.use_colors else ""
            lines.append(f"  {other_color}└─ Other:{reset_code}")
            # Indent each JSON line
            json_lines = json_str.split("\n")
            for line in json_lines:
                lines.append(f"    {other_color}{line}{reset_code}")
        
        if lines:
            # Add top border
            border_color = "\033[90m" if self.use_colors else ""
            result = f"\n{border_color}┌─ Metadata ─────────────────────────────────────{reset_code}\n"
            result += "\n".join(lines)
            result += f"\n{border_color}└────────────────────────────────────────────────{reset_code}"
            return result
        
        return ""
    
    async def handle(self, message: StructuredMessage):
        """
        Handle a structured message by outputting to the console (async).
        
        Uses a global lock to prevent output interleaving when multiple emitters
        print concurrently.
        
        Args:
            message: Structured message object
        """
        # Skip types that are only logged to file
        if message.type in (MessageType.TOOL_SCHEMA, MessageType.CONVERSATION_HISTORY):
            return
        
        # Extract basic info
        agent_name = message.context.get("agent_name", "unknown")
        iteration = message.context.get("iteration", 0)
        
        # Map type to display string
        type_display = self._get_type_display(message.type)
        if type_display is None:
            # For unsupported types, use the raw type name
            type_display = message.type.replace("_", " ")
        
        # Format content
        content = self._format_content(message)
        
        # Build output (with colors and fancy borders)
        agent_color = self._get_agent_color(agent_name)
        type_color = self._get_type_color(message.type)
        reset_code = self._get_reset_code()
        border_color = "\033[90m" if self.use_colors else ""  # gray border
        
        # Build separator line (using double-line chars for visibility)
        separator = "━" * 80
        
        # Build header and content
        header = f"{agent_color}[{agent_name}]{reset_code} [{iteration}] {type_color}[{type_display}]{reset_code}"
        content_lines = content.split("\n") if content else []
        
        # Build complete output (with borders)
        output_lines = [
            f"{border_color}{separator}{reset_code}",
            header,
        ]
        # Add content lines (each prefixed with a border character)
        if content_lines:
            output_lines.extend([f"│ {line}" for line in content_lines])
        output_lines.extend([
            f"{border_color}{separator}{reset_code}",
            ""  # trailing newline
        ])
        output = "\n".join(output_lines)
        
        # Output
        async with _console_lock:
            # Errors go to stderr, everything else to stdout
            output_stream = sys.stderr if message.type in (MessageType.ERROR, MessageType.TASK_FAILED) else sys.stdout
            print(output, file=output_stream, flush=True)
