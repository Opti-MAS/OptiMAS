"""OpenAI LLM implementation"""
import os
import random
import httpx
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
from openai import AsyncOpenAI

from .base import BaseLLM
from .usage_recorder import APIUsageRecord, APIUsageRecorder
from ..utils.exceptions import LLMError

# Matches factory.LOCAL_MODEL_PREFIX: this prefix routes to the local URL; stripped before backend requests
LOCAL_MODEL_PREFIX = "local_"


def merge_assistant_message_content(message: Any) -> str:
    """Build the assistant string stored in chat history.

    Some providers expose chain-of-thought or reasoning outside ``message.content``.
    When present, prepend it so **one** assistant turn carries both planning and answer
    (no separate ``think`` tool required for the model to leave a visible trace).
    """
    main = ""
    if hasattr(message, "content"):
        c = message.content
        if isinstance(c, str):
            main = c or ""
        elif isinstance(c, list):
            parts: List[str] = []
            for p in c:
                if isinstance(p, dict):
                    typ = (p.get("type") or "").lower()
                    txt = p.get("text") or ""
                    if typ in ("reasoning", "thought", "thinking", "internal_thought") and txt:
                        parts.append(f"### Reasoning\n{txt}")
                    elif txt:
                        parts.append(txt)
                elif isinstance(p, str) and p:
                    parts.append(p)
            main = "\n\n".join(parts)
        elif c is not None:
            main = str(c)

    sidecar = ""
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(message, attr, None)
        if isinstance(v, str) and v.strip():
            sidecar = v.strip()
            break
    if not sidecar and hasattr(message, "model_extra") and isinstance(message.model_extra, dict):
        for key in ("reasoning_content", "reasoning", "thinking"):
            v = message.model_extra.get(key)
            if isinstance(v, str) and v.strip():
                sidecar = v.strip()
                break

    if sidecar:
        if main.strip():
            return f"### Reasoning\n{sidecar}\n\n{main}".strip()
        return f"### Reasoning\n{sidecar}".strip()
    return main


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def _get_api_credentials(self) -> Tuple[str, Optional[str]]:
        """
        Retrieve the API key and base_url.

        For local models, the base_url is constructed from the config port with this priority:
        - config.base_url > config.port > environment variables > defaults
        - Models prefixed with local_ default to the local service address
        """
        if self.model_name.startswith(LOCAL_MODEL_PREFIX):
            config_base_url = self.config.get("base_url")
            config_port = self.config.get("port")
            api_key = self.config.get("api_key") or os.getenv("LOCAL_LLM_API_KEY", "local")
            if config_base_url:
                return api_key, config_base_url
            if config_port is not None:
                return api_key, f"http://127.0.0.1:{int(config_port)}/v1"
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
            return api_key, base_url
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or api_key in config.")
        base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        return api_key, base_url
    
    def _make_http_client(self) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with shared timeout / pool settings."""
        verify_ssl = os.getenv("LLM_VERIFY_SSL", "true").lower() in ("true", "1")
        read_timeout = self.config.get("read_timeout", 600.0)
        return httpx.AsyncClient(
            verify=verify_ssl,
            timeout=httpx.Timeout(
                connect=10.0,
                read=read_timeout,
                write=60.0,
                pool=60.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=30,
            ),
        )

    def _make_openai_client(self, base_url: Optional[str]) -> AsyncOpenAI:
        """Create an AsyncOpenAI client for *base_url*."""
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "http_client": self._make_http_client(),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    def __init__(self, model_name: str, api_usage_recorder: APIUsageRecorder, config: Dict[str, Any], global_llm_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model_name: Model name (e.g. "gpt-4", "gpt-3.5-turbo")
            api_usage_recorder: API usage recorder
            config: Configuration dict (max_retries, etc.)
            global_llm_settings: Global LLM settings (optional); merged into config in BaseLLM
        """
        super().__init__(model_name, api_usage_recorder=api_usage_recorder, config=config, global_llm_settings=global_llm_settings)
        
        self._api_model_name = model_name[len(LOCAL_MODEL_PREFIX):] if model_name.startswith(LOCAL_MODEL_PREFIX) else model_name
        self._active_api_model_name = self._api_model_name
        
        self.api_key, base_url = self._get_api_credentials()
        
        # Primary client
        self.client = self._make_openai_client(base_url)
        self.base_url = base_url

        # Load-balancing pool: [(AsyncOpenAI, base_url), ...]
        # Requests are randomly distributed across all clients.
        self._client_pool: List[Tuple[AsyncOpenAI, Optional[str]]] = [(self.client, base_url)]
        extra_urls: List[str] = self.config.get("extra_base_urls") or []
        for url in extra_urls:
            self._client_pool.append((self._make_openai_client(url), url))
    
    @staticmethod
    def _strip_anyof(obj: Any) -> Any:
        """Recursively simplify ``anyOf`` produced by Optional types.

        ``Optional[str]`` generates ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``.
        OpenAI strict-mode rejects ``anyOf``, so we collapse it to the first
        non-null variant (preserving description / enum if present).
        """
        if isinstance(obj, dict):
            if "anyOf" in obj and isinstance(obj["anyOf"], list):
                variants = [v for v in obj["anyOf"] if not (isinstance(v, dict) and v.get("type") == "null")]
                base = variants[0] if variants else obj["anyOf"][0]
                merged = {k: v for k, v in obj.items() if k != "anyOf"}
                if isinstance(base, dict):
                    for k, v in base.items():
                        merged.setdefault(k, v)
                return {k: OpenAILLM._strip_anyof(v) for k, v in merged.items()}
            return {k: OpenAILLM._strip_anyof(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [OpenAILLM._strip_anyof(item) for item in obj]
        return obj

    def _convert_tool_schemas_to_openai_format(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert fastmcp-format tool schemas to OpenAI function calling format.
        Automatically cleans up: ensures additionalProperties: false, simplifies anyOf.
        
        Args:
            schemas: fastmcp-format tool schema list ({"name": str, "description": str, "inputSchema": dict})
        
        Returns:
            Tool list in OpenAI function calling format
        """
        openai_tools = []
        for schema in schemas:
            if "inputSchema" in schema:
                parameters = self._strip_anyof(schema.get("inputSchema", {}))
                if "additionalProperties" not in parameters:
                    parameters["additionalProperties"] = False
                
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": schema.get("name"),
                        "description": schema.get("description", ""),
                        "parameters": parameters
                    }
                }
                openai_tools.append(openai_tool)
            else:
                if schema.get("type") == "function" and "function" in schema and "parameters" in schema["function"]:
                    parameters = self._strip_anyof(schema["function"]["parameters"])
                    if "additionalProperties" not in parameters:
                        parameters["additionalProperties"] = False
                    schema["function"]["parameters"] = parameters
                openai_tools.append(schema)
        return openai_tools
    
    def _swap_model(self, new_model_name: str) -> None:
        """Switch the model sent in API requests."""
        super()._swap_model(new_model_name)
        self._active_api_model_name = (
            new_model_name[len(LOCAL_MODEL_PREFIX):]
            if new_model_name.startswith(LOCAL_MODEL_PREFIX)
            else new_model_name
        )

    async def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Asynchronously call the OpenAI API and return the raw response object.
        
        Args:
            messages: Message list (OpenAI chat.completions format)
            **kwargs: API call parameters (tools, temperature, max_tokens, etc.)
        
        Returns:
            OpenAI chat.completions response object
        """
        api_params = {
            "model": self._active_api_model_name,
            "messages": messages,
        }
        for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "extra_body"):
            if key in self.config:
                api_params[key] = self.config[key]
        
        if "tools" in kwargs and kwargs["tools"]:
            api_params["tools"] = self._convert_tool_schemas_to_openai_format(kwargs["tools"])
            kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
        
        api_params.update(kwargs)
        
        # Pick a random client from the load-balancing pool
        client, _ = random.choice(self._client_pool)
        return await client.chat.completions.create(**api_params)
    
    def extract_chat_completion_response(self, api_response: Any, tools: Optional[List[Dict[str, Any]]] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Extract a unified response from the OpenAI API response.
        
        If api_response is None, validation fails, or extraction fails, returns (None, error_message)
        instead of raising an exception. Validates the response internally first.
        
        Args:
            api_response: OpenAI chat.completions response object (may be None)
            tools: Tool definition list (for validating tool calls, optional)
            messages: Message history (for detecting duplicate tool calls, optional)
        
        Returns:
            Tuple[Optional[Dict[str, Any]], str]: (extracted_response, error_message)
            - extracted_response: Unified response dict; None if validation or extraction fails
            - error_message: Error message (if failed, otherwise empty string)
        """
        if api_response is None:
            return None, "Response is None"
        
        try:
            # Validate response first (using base class validation)
            valid, error_msg = self._verify_response(api_response, tools, messages)
            if not valid:
                return None, error_msg
            
            # Extract response content
            choice = api_response.choices[0]
            message = choice.message
            
            # Text content (merges reasoning / multi-part content returned separately by some models)
            text_content = merge_assistant_message_content(message)
            
            # Extract tool calls
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
                    extras = getattr(tool_call, "model_extra", None) or {}
                    for key in ("extra_content", "thought_signature"):
                        if key in extras:
                            tool_calls[-1][key] = extras[key]
            
            # Extract metadata
            metadata = {
                "model": api_response.model if hasattr(api_response, 'model') else "",
                "usage": {
                    "prompt_tokens": api_response.usage.prompt_tokens,
                    "completion_tokens": api_response.usage.completion_tokens,
                    "total_tokens": api_response.usage.total_tokens,
                } if hasattr(api_response, 'usage') and api_response.usage else {},
                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None,
            }
            
            return {
                "text_content": text_content,
                "tool_calls": tool_calls,
                "reasoning_summary": "",  # Reasoning is merged into text_content where possible; field kept for telemetry
                "metadata": metadata
            }, ""
        except Exception as e:
            # On extraction failure, return None and error message instead of raising
            return None, f"Failed to extract response: {str(e)}"
    
    def _create_usage_record(
        self,
        messages: List[Dict[str, Any]],
        response: Optional[Dict[str, Any]],
        success: bool,
        error_message: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        **kwargs
    ) -> APIUsageRecord:
        """
        Create an API usage record.
        
        Args:
            messages: Input message list
            response: API response (on success) or None (on failure)
            success: Whether the call succeeded
            error_message: Error message (on failure)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total number of tokens
            **kwargs: Additional parameters
        
        Returns:
            APIUsageRecord instance
        """
        # Token info is passed in via parameters; use directly
        # Build outputs: remove metadata from response to avoid duplication with top-level metadata
        outputs = []
        if success and response:
            # Copy response but remove metadata (top-level already has complete metadata)
            output_response = response.copy()
            output_response.pop("metadata", None)
            outputs = [output_response]
        
        
        # Build metadata
        metadata = {}
        if success and response:
            metadata = response.get("metadata", {}).copy()
        
        # Add API call timing info to metadata
        if 'api_call_start_time' in kwargs:
            metadata['api_call_start_time'] = kwargs['api_call_start_time']
        if 'api_call_duration' in kwargs:
            metadata['api_call_duration'] = kwargs['api_call_duration']
        
        return APIUsageRecord(
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            success=success,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            base_url=self.base_url,
            inputs=messages,
            outputs=outputs,
            metadata=metadata
        )
