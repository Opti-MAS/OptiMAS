"""Gemini 3 LLM implementation"""
import os
from typing import List, Dict, Any, Optional, Tuple

from .openai_llm import OpenAILLM
from ..utils.exceptions import LLMError


class GeminiLLM(OpenAILLM):
    """Gemini LLM implementation, built on the OpenAI SDK"""
    
    def _get_api_credentials(self) -> Tuple[str, Optional[str]]:
        """
        Retrieve Gemini API key and base_url from environment variables.
        
        Returns:
            Tuple[str, Optional[str]]: (api_key, base_url)
        
        Raises:
            LLMError: If the API key is not provided
        """
        api_key = self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
        base_url = self.config.get("base_url") or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
        return api_key, base_url
    
    def __init__(self, model_name: str, api_usage_recorder, config: Dict[str, Any], global_llm_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the Gemini LLM.
        
        Args:
            model_name: Model name (e.g. "gemini-3-pro-preview")
            api_usage_recorder: API usage recorder
            config: Configuration dict (max_retries, etc.)
            global_llm_settings: Global LLM settings (optional); merged into config in BaseLLM
        """
        # Use the provider's model defaults unless generation settings are supplied.
        config = config.copy() if config else {}
        
        # Call parent class initializer
        super().__init__(model_name, api_usage_recorder=api_usage_recorder, config=config, global_llm_settings=global_llm_settings)
    
    async def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Asynchronously call the Gemini API and return the raw response object.
        Uses the OpenAI SDK with Gemini-specific configuration.
        
        Args:
            messages: Message list (OpenAI chat.completions format)
            **kwargs: API call parameters (tools, temperature, max_tokens, etc.)
        
        Returns:
            OpenAI chat.completions response object
        """
        
        # Prepare API parameters
        api_params = {
            "model": self._active_api_model_name,
            "messages": messages,
        }
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "reasoning_effort"):
            if key in self.config:
                api_params[key] = self.config[key]
        
        # Handle tools parameter: convert from FastMCP tool schemas
        if "tools" in kwargs and kwargs["tools"]:
            api_params["tools"] = self._convert_tool_schemas_to_openai_format(kwargs["tools"])
            # Remove tools from kwargs to avoid duplication
            kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        
        # Add remaining parameters (all kwargs)
        api_params.update(kwargs)
        
        # Merge extra_body from config (defaults set in __init__)
        if "extra_body" in self.config:
            extra_body = api_params.setdefault("extra_body", {})
            extra_body.update(self.config["extra_body"])
        
        # Set default tool_choice
        if "tools" in api_params and "tool_choice" not in api_params:
            api_params["tool_choice"] = "auto"
        
        # Async call to OpenAI API (Gemini-compatible)
        response = await self.client.chat.completions.create(**api_params)
        return response
    
    def extract_chat_completion_response(self, api_response: Any, tools: Optional[List[Dict[str, Any]]] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """Preserve provider tool-call metadata, including real thought signatures."""
        return super().extract_chat_completion_response(api_response, tools, messages)
