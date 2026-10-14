"""
Custom exceptions for OptiMAS Server.
"""


class OptiMASServerError(Exception):
    """Base exception for OptiMAS Server"""

    pass


class SessionError(OptiMASServerError):
    """Session-related errors"""

    pass


class SessionNotFoundError(SessionError):
    """Session not found"""

    pass


class SessionExpiredError(SessionError):
    """Session expired"""

    pass


class SessionLimitExceededError(SessionError):
    """Maximum number of sessions exceeded"""

    pass


class ConnectionError(OptiMASServerError):
    """Connection-related errors"""

    pass


class AuthenticationError(OptiMASServerError):
    """Authentication errors"""

    pass


class ConfigurationError(OptiMASServerError):
    """Configuration errors"""

    pass

