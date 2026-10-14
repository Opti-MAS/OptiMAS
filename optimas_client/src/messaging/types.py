"""Message type definitions and structured message class"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any


# Message type constants
class MessageType:
    """Message type constants class"""
    
    # Base message types
    WARNING = "warning"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE = "message"
    USER_PROMPT = "user_prompt"
    SYSTEM_PROMPT = "system_prompt"
    
    # Workflow message types
    TASK_START = "task_start"
    TASK_DONE = "task_done"
    TASK_FAILED = "task_failed"
    PHASE_SUMMARY = "phase_summary"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    ITERATION_START = "iteration_start"
    TOOL_SCHEMA = "tool_schema"  # Tool schema info (logged to file only)
    CONVERSATION_HISTORY = "conversation_history"  # Conversation history (logged to file only)


@dataclass
class StructuredMessage:
    """Structured message class
    
    Contains all message information for unified handler processing.
    """
    type: str  # Message type (using MessageType constants)
    content: str  # Message content
    metadata: Dict[str, Any]  # Metadata (optional fields)
    context: Dict[str, Any]  # Context info (agent_name, task_id, phase_count)
    timestamp: datetime  # Timestamp
    
    def __post_init__(self):
        """Ensure metadata and context are not None"""
        if self.metadata is None:
            self.metadata = {}
        if self.context is None:
            self.context = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for serialization)"""
        return {
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }
