"""Custom exceptions"""


class OptiMASClientError(Exception):
    """OptiMAS Client base exception"""
    pass


class LLMError(OptiMASClientError):
    """LLM-related error"""
    pass


class ContextOverflowError(LLMError):
    """Raised when the conversation exceeds the model's context window.

    This is NOT retryable by simply calling the API again — the caller
    must truncate / compact the conversation before retrying.
    """
    pass


class MCPError(OptiMASClientError):
    """MCP-related error"""
    pass


class AgentError(OptiMASClientError):
    """Agent-related error"""
    pass


class ConfigError(OptiMASClientError):
    """Configuration-related error"""
    pass


class MessagingError(OptiMASClientError):
    """Messaging-related error"""
    pass
