"""Abstract base class for message handlers"""
from abc import ABC, abstractmethod
from ..types import StructuredMessage


class BaseHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle(self, message: StructuredMessage):
        """
        Handle a structured message (async).
        
        Args:
            message: Structured message object
        """
        pass
