"""Message emitter"""
import asyncio
import inspect
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .handlers.base import BaseHandler
from .handlers.console import ConsoleHandler
from .handlers.file import TrajFileHandler
from .types import MessageType, StructuredMessage
from ..utils.exceptions import MessagingError

logger = logging.getLogger(__name__)


class MessageEmitter:
    """Message emitter with plugin-based handlers"""
    
    def __init__(self, handlers: Optional[List[BaseHandler]] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the message emitter.
        
        Args:
            handlers: List of message handlers; defaults to ConsoleHandler if None
            context: Context info (agent_name, task_id, phase_count)
        """
        if handlers is None:
            handlers = [ConsoleHandler()]
        
        self.handlers: List[BaseHandler] = list(handlers)
        self.context: Dict[str, Any] = context or {}
    
    def add_handler(self, handler: BaseHandler):
        """
        Add a message handler.
        
        Args:
            handler: Message handler instance
        """
        if not isinstance(handler, BaseHandler):
            raise MessagingError("Handler must be an instance of BaseHandler")
        self.handlers.append(handler)
    
    def remove_handler(self, handler: BaseHandler):
        """
        Remove a message handler.
        
        Args:
            handler: Handler instance to remove
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def set_context(self, agent_name: str, task_id: str, phase_count: int = 0,
                     invocation_id: str = ""):
        """Set context information.

        Args:
            invocation_id: 16-char hex string that uniquely identifies one
                ``execute_task`` invocation.  Used in trajectory filenames to
                disambiguate repeated calls to the same sub-agent name.
        """
        self.context = {
            "agent_name": agent_name,
            "task_id": task_id,
            "phase_count": phase_count,
            "invocation_id": invocation_id,
            "iteration": 0,
            "real_iteration": 0
        }
    
    def update_context(self, **kwargs):
        """Update context information."""
        self.context.update(kwargs)
    
    def update_iteration(self, iteration: int, real_iteration: int):
        """Update iteration count."""
        self.context["iteration"] = iteration
        self.context["real_iteration"] = real_iteration
    
    def increment_phase(self):
        """Increment the phase count."""
        self.context["phase_count"] = self.context.get("phase_count", 0) + 1
    
    async def emit(
        self,
        message_type: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit a structured message to all registered handlers (core async method).
        
        Args:
            message_type: Message type (using MessageType constants)
            content: Message content
            metadata: Metadata (optional)

        Returns:
            True if all handlers succeeded, False if any handler failed.
        """
        # Create structured message (context as a standalone field)
        structured_message = StructuredMessage(
            type=message_type,
            content=content,
            metadata=metadata or {},
            context=self.context.copy(),
            timestamp=datetime.now()
        )
        
        # Execute all handlers concurrently
        tasks = []
        for handler in self.handlers:
            if inspect.iscoroutinefunction(handler.handle):
                tasks.append(handler.handle(structured_message))
            else:
                # Wrap synchronous handlers as async (backward compatible)
                tasks.append(asyncio.to_thread(handler.handle, structured_message))
        
        # Wait for all handlers to complete with error isolation
        all_ok = True
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_ok = False
                logger.error(
                    "Handler %s failed for message_type=%s: %s",
                    type(self.handlers[i]).__name__, message_type, result,
                    exc_info=result,
                )
        return all_ok
    
    # ========== Core convenience methods ==========
    
    async def emit_tool(
        self,
        tool_name: str,
        tool_call_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a tool-related message (call or result).
        
        Args:
            tool_name: Tool name (required)
            tool_call_id: Tool call ID (required, used to pair calls and results)
            arguments: Tool arguments (optional, provided on call)
            success: Whether successful (optional, provided on result)
            data: Tool return data (optional, provided on result)
            error: Error message (optional, provided on failure)
            metadata: Additional metadata (optional)
        
        Examples:
            # Tool call
            emitter.emit_tool(tool_name="read_file", tool_call_id="call_123", arguments={"path": "file.txt"})
            
            # Tool result (success)
            emitter.emit_tool(tool_name="read_file", tool_call_id="call_123", success=True, data="file content")
            
            # Tool result (failure)
            emitter.emit_tool(tool_name="read_file", tool_call_id="call_123", success=False, error="File not found")
        """
        if metadata is None:
            metadata = {}
        metadata.update({
            "tool_name": tool_name,
            "tool_call_id": tool_call_id
        })
        
        if success is None:
            # Tool call
            if arguments is None:
                raise ValueError("arguments parameter is required when emitting tool call")
            metadata.update({
                "arguments": arguments
            })
            content = f"Calling tool: {tool_name}"
            message_type = MessageType.TOOL_CALL
        else:
            # Tool result
            if arguments is not None:
                metadata["arguments"] = arguments
            metadata.update({
                "success": success,
                "data": data,
                "error": error
            })
            if success:
                content = f"Tool {tool_name} succeeded"
            else:
                content = f"Tool {tool_name} failed: {error or 'Unknown error'}"
            message_type = MessageType.TOOL_RESULT
        
        await self.emit(message_type, content, metadata)
    
    async def emit_task(
        self,
        status: str,
        task: Optional[str] = None,
        summary: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a task-related message (start/done/failed).
        
        Args:
            status: Task status, must be one of "start", "done", or "failed" (required)
            task: Task description (required when status is "start")
            summary: Task summary (required when status is "done")
            error: Error message (required when status is "failed")
            metadata: Additional metadata (optional)
        
        Examples:
            # Task start
            emitter.emit_task(status="start", task="Build website")
            
            # Task done
            emitter.emit_task(status="done", summary="Website built successfully")
            
            # Task failed
            emitter.emit_task(status="failed", error="Build timeout")
        """
        if metadata is None:
            metadata = {}
        
        if status == "start":
            if task is None:
                raise ValueError("task parameter is required when status is 'start'")
            metadata.update({"task": task})
            content = f"Task started: {task}"
            message_type = MessageType.TASK_START
        elif status == "done":
            if summary is None:
                raise ValueError("summary parameter is required when status is 'done'")
            metadata.update({"summary": summary})
            content = f"Task completed: {summary}"
            message_type = MessageType.TASK_DONE
        elif status == "failed":
            if error is None:
                raise ValueError("error parameter is required when status is 'failed'")
            metadata.update({"error": error})
            content = f"Task failed: {error}"
            message_type = MessageType.TASK_FAILED
        else:
            raise ValueError(f"status must be 'start', 'done', or 'failed', got '{status}'")
        
        await self.emit(message_type, content, metadata)
    
    async def emit_phase(
        self,
        progress_report: str,
        remaining_tasks: str,
        iteration: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Emit a phase summary message.
        
        Args:
            progress_report: Progress report (required)
            remaining_tasks: Remaining tasks (required)
            iteration: Current iteration number (required)
            metadata: Additional metadata (optional)
        
        Examples:
            emitter.emit_phase(
                progress_report="Completed UI design",
                remaining_tasks="Implement backend API",
                iteration=5
            )
        """
        if metadata is None:
            metadata = {}
        metadata.update({
            "progress_report": progress_report,
            "remaining_tasks": remaining_tasks,
            "iteration": iteration
        })
        content = f"Phase summary (iteration {iteration})"
        await self.emit(MessageType.PHASE_SUMMARY, content, metadata)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> "MessageEmitter":
        """
        Create a MessageEmitter from configuration.
        
        Args:
            config: Configuration dictionary in the format:
                {
                    "handlers": [
                        {"type": "console"},
                        {"type": "file", "base_path": "trajs"}  # base_path is optional, defaults to "trajs"
                    ]
                }
            context: Optional context info dictionary
        
        Returns:
            A MessageEmitter instance
        """
        handlers = []
        handlers_config = config.get("handlers", [])
        
        for handler_config in handlers_config:
            handler_type = handler_config.get("type", "").lower()
            
            if handler_type == "console":
                use_colors = handler_config.get("use_colors", True)
                handlers.append(ConsoleHandler(use_colors=use_colors))
            elif handler_type == "file":
                # TrajFileHandler uses dynamic paths extracted from context
                base_path = handler_config.get("base_path", "trajs")
                handlers.append(TrajFileHandler(base_path=base_path))
            else:
                raise MessagingError(f"Unknown handler type: {handler_type}")
        
        # If no handlers configured, use default ConsoleHandler
        if not handlers:
            handlers = [ConsoleHandler()]
        
        return cls(handlers=handlers, context=context)
