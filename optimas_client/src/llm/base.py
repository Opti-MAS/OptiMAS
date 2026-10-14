"""LLM abstract base class"""
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy
from datetime import datetime

from .usage_recorder import APIUsageRecorder, APIUsageRecord
from .llm_utils import handle_exception, is_server_side_error
from ..utils.exceptions import LLMError, ContextOverflowError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model fallback registry (populated from model_configs.yaml at load time)
# ---------------------------------------------------------------------------
_model_fallback_registry: Optional[Dict[str, list]] = None


def _load_fallback_registry() -> Dict[str, list]:
    """Load model fallback chains from model_configs.yaml (cached)."""
    global _model_fallback_registry
    if _model_fallback_registry is not None:
        return _model_fallback_registry

    from pathlib import Path
    import yaml as _yaml

    cfg_path = Path(__file__).parent.parent.parent / "config" / "model_configs.yaml"
    registry: Dict[str, list] = {}
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f) or {}
            for name, entry in (data.get("models") or {}).items():
                fb = entry.get("fallbacks")
                if isinstance(fb, list) and fb:
                    registry[name] = list(fb)
        except Exception:
            pass
    _model_fallback_registry = registry
    return registry


def get_model_fallbacks(model_name: str) -> list:
    """Return the ordered fallback list for *model_name* (empty if none)."""
    return list(_load_fallback_registry().get(model_name, []))

# ---------------------------------------------------------------------------
# Global LLM concurrency limiter
# ---------------------------------------------------------------------------
# All BaseLLM.generate() calls in the process share this semaphore.
# Prevents overloading local LLM servers when multiple agents run concurrently
# (each agent has its own LLM client, but the backend GPU is shared).
# Value is lazily initialised on first use and can be overridden via config.

_global_llm_semaphore: Optional[asyncio.Semaphore] = None
_global_llm_max_concurrency: int = 0  # 0 = not yet initialised


def configure_global_llm_concurrency(max_concurrent: int) -> None:
    """Set the process-wide LLM concurrency limit.

    Must be called *before* the first ``generate()`` call (typically in train.py
    or the forward engine setup).  Calling again after first use logs a warning
    but updates the limit for new requests.
    """
    global _global_llm_semaphore, _global_llm_max_concurrency
    if max_concurrent <= 0:
        _global_llm_semaphore = None
        _global_llm_max_concurrency = 0
        return
    _global_llm_max_concurrency = max_concurrent
    _global_llm_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info("Global LLM concurrency limit set to %d", max_concurrent)


def _get_global_llm_semaphore() -> Optional[asyncio.Semaphore]:
    """Return the global semaphore, lazily creating a default if not configured."""
    global _global_llm_semaphore, _global_llm_max_concurrency
    if _global_llm_max_concurrency == 0:
        # Auto-configure with a sensible default for local LLM deployments
        _global_llm_max_concurrency = 100
        _global_llm_semaphore = asyncio.Semaphore(_global_llm_max_concurrency)
        logger.info(
            "Auto-configured global LLM concurrency limit to %d "
            "(call configure_global_llm_concurrency() to override)",
            _global_llm_max_concurrency,
        )
    return _global_llm_semaphore


class ResponseValidationError(LLMError):
    """Exception raised when response validation fails"""
    def __init__(self, message: str, response: Any = None):
        super().__init__(message)
        self.message = message
        self.response = response


class BaseLLM(ABC):
    """LLM abstract base class"""
    
    def __init__(self, model_name: str, api_usage_recorder: APIUsageRecorder, config: Dict[str, Any], global_llm_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM.
        
        Args:
            model_name: Model name
            api_usage_recorder: API usage recorder
            config: Configuration dict (e.g. max_retries, passed through to subclasses)
            global_llm_settings: Global LLM settings (optional); merged into config if provided
        """
        self.model_name = model_name
        self.api_usage_recorder = api_usage_recorder
        
        # Merge global settings into config (if provided)
        # Copy config to avoid mutating the original dict
        merged_config = config.copy() if config else {}
        
        if global_llm_settings:
            # Merge all global settings; LLM-specific config takes precedence
            # Apply global settings first, then overlay LLM config (higher priority)
            temp_config = global_llm_settings.copy()
            temp_config.update(merged_config)
            merged_config = temp_config
        
        self.config = merged_config
        self.max_retries = self.config.get("max_retries", 3)

        self._fallback_models: list = get_model_fallbacks(model_name)
        self._active_model_name: str = model_name
    
    def _swap_model(self, new_model_name: str) -> None:
        """Switch the model used for API calls (subclass may override for deeper state)."""
        self._active_model_name = new_model_name

    @abstractmethod
    async def _call_api(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Call the LLM API and return the raw response object (must be implemented by subclasses).
        
        Args:
            messages: List of messages
            **kwargs: API call parameters
        
        Returns:
            Raw API response object
        """
        pass
    
    @abstractmethod
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
        Create an API usage record (must be implemented by subclasses).
        
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
        pass
    
    @abstractmethod
    def extract_chat_completion_response(self, api_response: Any, tools: Optional[List[Dict[str, Any]]] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Extract a unified response from the raw API response (must be implemented by subclasses).
        
        If api_response is None, validation fails, or extraction fails, return (None, error_message)
        instead of raising an exception. Subclasses should validate the response first and return
        (None, error_message) on validation failure. Exception handling should be done internally.
        
        Args:
            api_response: Raw API response object (may be None)
            tools: Tool definition list (for validating tool calls, optional)
            messages: Message history (for detecting duplicate tool calls, optional)
        
        Returns:
            Tuple[Optional[Dict[str, Any]], str]: (extracted_response, error_message)
            - extracted_response: Extracted response dict containing:
              - text_content: str - Text content
              - tool_calls: List[Dict] - Tool call list (if any)
              - reasoning_summary: str - Reasoning summary (if any)
              - metadata: Dict - Metadata (token usage, etc.)
              Returns None if validation or extraction fails
            - error_message: Error message (if extraction fails, otherwise empty string)
        """
        pass
    
    def _verify_inputs(self, messages: List[Dict[str, Any]], **kwargs) -> bool:
        """
        Validate the input message format.
        
        Args:
            messages: List of messages
            **kwargs: Additional parameters
        
        Returns:
            bool: True if the input is valid, False otherwise
        """
        if not isinstance(messages, list):
            return False
        
        if len(messages) == 0:
            return False
        
        valid_roles = {"system", "user", "assistant", "tool"}
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            
            role = msg.get("role")
            if role not in valid_roles:
                return False
            
            # Check the content field (tool messages must have tool_call_id)
            if role == "tool":
                if "tool_call_id" not in msg:
                    return False
            else:
                if "content" not in msg:
                    return False
        
        return True
    
    def _verify_response(self, api_response: Any, tools: Optional[List[Dict[str, Any]]] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, str]:
        """
        Validate whether the API response is valid.
        
        The default implementation expects OpenAI chat.completions format.
        Subclasses may override this method to support other formats.
        
        Args:
            api_response: Raw API response object
            tools: Tool definition list (for validating tool calls)
            messages: Message history (for detecting duplicate tool calls)
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if api_response is None:
            return False, "Response is None"
        
        # Check response structure — expects OpenAI chat.completions format
        if not hasattr(api_response, 'choices') or not api_response.choices:
            return False, "Response missing 'choices' field or choices is empty"
        
        choice = api_response.choices[0]
        if not hasattr(choice, 'message'):
            return False, "Response choice missing 'message' field"
        
        message = choice.message
        
        # Check for tool_calls or content
        has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
        has_content = hasattr(message, 'content') and message.content
        
        # Invalid if neither tool_calls nor content is present
        if not has_tool_calls and not has_content:
            return False, "Response message has neither tool_calls nor content"
        
        # If no tool_calls, the response is valid
        if not has_tool_calls:
            return True, ""
        
        # Validate tool_calls
        if not tools:
            return False, "Response contains tool_calls but no tools provided"
        
        # Build tool name mapping
        tool_map = {}
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get('type') == 'function':
                    tool_name = tool.get('function', {}).get('name', '')
                else:
                    tool_name = tool.get('name', '')
                if tool_name:
                    tool_map[tool_name] = tool
        
        # Validate each tool_call
        for tool_call in message.tool_calls:
            if not hasattr(tool_call, 'function'):
                return False, "Tool call missing 'function' field"
            
            function = tool_call.function
            if not hasattr(function, 'name'):
                return False, "Tool call function missing 'name' field"
            
            tool_name = function.name
            
            # Check if tool name is valid
            if tool_name not in tool_map:
                valid_tool_names = list(tool_map.keys())
                return False, f"Invalid tool name '{tool_name}'. Valid tools: {valid_tool_names}"
            
            # Check if arguments is a valid JSON string
            if not hasattr(function, 'arguments'):
                return False, f"Tool call function '{tool_name}' missing 'arguments' field"
            
            arguments_str = function.arguments
            if not isinstance(arguments_str, str):
                return False, f"Tool call arguments for '{tool_name}' must be a JSON string, got {type(arguments_str)}"
            
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON in function_call arguments for '{tool_name}': {str(e)}"
            except Exception as e:
                return False, f"Error parsing arguments for '{tool_name}': {str(e)}"
        
        # Detect duplicate tool calls: compare current and previous tool calls
        if messages and has_tool_calls:
            # Extract tool calls from the last assistant message in history
            last_tool_calls = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_tool_calls = msg.get("tool_calls")
                    break
            
            if last_tool_calls:
                # Helper: generate tool call signature (tool name + sorted argument JSON)
                def make_signature(tool_name: str, arguments_str: str) -> Optional[tuple]:
                    try:
                        arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                        return (tool_name, json.dumps(arguments, sort_keys=True))
                    except Exception:
                        return None
                
                # Build signature list for current tool calls
                current_sigs = [
                    make_signature(tc.function.name, tc.function.arguments)
                    for tc in message.tool_calls
                ]
                current_sigs = [s for s in current_sigs if s is not None]
                
                # Build signature list for previous tool calls
                last_sigs = [
                    make_signature(tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments", "{}"))
                    for tc in last_tool_calls
                ]
                last_sigs = [s for s in last_sigs if s is not None]
                
                # If current tool calls are identical to the previous ones, return error
                if current_sigs == last_sigs and current_sigs:
                    return False, "Duplicate tool calls detected: This response contains the same tool calls as the previous response. Please try a different approach."
        
        return True, ""
    
    def _clean_messages_for_api(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean the message list by removing internal fields (underscore-prefixed keys) before sending to the LLM API.
        
        Args:
            messages: Original message list
        
        Returns:
            Cleaned message list (deep copy; does not modify the originals)
        """
        cleaned_messages = []
        for msg in messages:
            # Deep copy messages to avoid mutating the originals
            cleaned_msg = deepcopy({k: v for k, v in msg.items() if not k.startswith("_")})
            cleaned_messages.append(cleaned_msg)
        return cleaned_messages
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        retry: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an LLM response (with input validation, retry logic, and error handling).
        
        Args:
            messages: Message history in OpenAI chat.completions format
                     [{"role": "system/user/assistant/tool", "content": "...", ...}]
            retry: Number of retries (optional; defaults to max_retries)
            **kwargs: Additional generation parameters (temperature, max_tokens, tools, etc.)
        
        Returns:
            Dict containing:
            - text_content: str - Text content
            - tool_calls: List[Dict] - Tool call list (if any)
            - reasoning_summary: str - Reasoning summary (if any)
            - metadata: Dict - Metadata (token usage, etc.)
        
        Raises:
            LLMError: If input is invalid or all retries are exhausted
            ResponseValidationError: If response validation fails
        """
        max_attempts = retry if retry is not None else self.max_retries
        max_attempts = max(1, int(max_attempts))
        
        # Validate input
        if not self._verify_inputs(messages, **kwargs):
            raise LLMError("Inputs are not valid, please check the inputs and try again")
        
        error_messages: List[str] = []  # Detailed error info for the final exception
        llm_error_messages: List[str] = []  # Error hints passed to the LLM for retry
        attempt = 0

        # Model-not-found fallback state
        fallback_chain = [self.model_name] + list(self._fallback_models)
        fallback_idx = 0  # index into fallback_chain; 0 = primary model
        
        while attempt < max_attempts:
            # Initialize variables (used in the finally block)
            api_response = extracted_response = None
            success = False
            error_message = ""
            api_call_start_time_str = api_call_start_timestamp = api_call_duration = None

            # If there are previous error messages, append them to prompt the LLM
            messages_with_retry_info = deepcopy(messages)
            if llm_error_messages:
                last_llm_error = llm_error_messages[-1]
                if last_llm_error:  # Only append non-empty error messages
                    if messages_with_retry_info[-1].get("role") != "user":
                        messages_with_retry_info.append({
                            "role": "user",
                            "content": f"Last generate failed: {last_llm_error}",
                        })
                    else:
                        messages_with_retry_info[-1]["content"] += f"\n\nLast generate failed: {last_llm_error}"
            
            # Clean messages by removing internal fields (e.g. _llm_metadata) before sending to the LLM API
            cleaned_messages = self._clean_messages_for_api(messages_with_retry_info)
            
            try:
                # Deep copy kwargs to prevent mutation by the SDK
                api_kwargs = deepcopy(kwargs)
                
                # Record API call start time
                api_call_start_timestamp = time.time()
                api_call_start_time_str = datetime.fromtimestamp(api_call_start_timestamp).strftime("%Y-%m-%d_%H-%M-%S")
                
                # Acquire global concurrency semaphore to prevent overloading
                # the backend LLM server (especially important for local deployments)
                sem = _get_global_llm_semaphore()
                if sem is not None:
                    await sem.acquire()
                try:
                    api_response = await self._call_api(cleaned_messages, **api_kwargs)
                finally:
                    if sem is not None:
                        sem.release()
                
                # Calculate API call duration
                api_call_duration = time.time() - api_call_start_timestamp
                
                # Extract unified response (validation happens internally)
                extracted_response, error_msg = self.extract_chat_completion_response(api_response, kwargs.get('tools'), messages_with_retry_info)
                if extracted_response is None:
                    raise ResponseValidationError(error_msg or "Failed to extract response from API response", response=api_response)
                
                # Add duration to metadata (for saving to conversation_history)
                if api_call_duration is not None:
                    if "metadata" not in extracted_response:
                        extracted_response["metadata"] = {}
                    extracted_response["metadata"]["duration"] = api_call_duration
                
                # On success, restore primary model for next call if we were on a fallback
                if fallback_idx != 0:
                    logger.info(
                        "Request succeeded on fallback model '%s'; "
                        "will try primary '%s' again on next call.",
                        self._active_model_name, self.model_name,
                    )
                    self._swap_model(self.model_name)
                    fallback_idx = 0

                success = True
                return extracted_response
                
            except (ResponseValidationError, Exception) as e:
                extracted_response, _ = self.extract_chat_completion_response(api_response, kwargs.get('tools'), messages_with_retry_info)

                # ContextOverflowError is NOT retryable — let it propagate
                # immediately so the agent layer can compact context.
                # handle_exception will raise ContextOverflowError internally
                # when it detects context-overflow patterns.

                # ── Server-side error fallback: try next model variant before
                # counting this as a full retry attempt ─────────────────────
                if is_server_side_error(e) and len(fallback_chain) > 1:
                    fallback_idx += 1
                    if fallback_idx < len(fallback_chain):
                        next_model = fallback_chain[fallback_idx]
                        logger.warning(
                            "Server error on '%s' (attempt %d, %s); "
                            "falling back to '%s' (%d/%d in chain).",
                            self._active_model_name, attempt + 1,
                            type(e).__name__,
                            next_model, fallback_idx + 1, len(fallback_chain),
                        )
                        self._swap_model(next_model)
                        await asyncio.sleep(2)
                        continue
                    else:
                        fallback_idx = 0
                        self._swap_model(fallback_chain[0])
                        logger.warning(
                            "All %d model variants failed; "
                            "counting as 1 retry attempt (%d/%d).",
                            len(fallback_chain), attempt + 1, max_attempts,
                        )

                try:
                    error_message, detailed_error, attempt_increment = await handle_exception(e, api_response)
                except ContextOverflowError as overflow_err:
                    # Record the failed call before re-raising (use overflow_err, not outer *e*)
                    error_message = str(overflow_err)
                    success = False
                    raise

                logger.warning("LLM attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e)

                error_messages.append(f"Attempt {attempt + 1}: {detailed_error}")
                llm_error_messages.append(error_message)
                attempt += attempt_increment
                continue
            
            finally:
                # Record API usage regardless of success or failure
                # If the call started but duration wasn't computed, calculate it now (for failure cases)
                if api_call_start_timestamp is not None and api_call_duration is None:
                    api_call_duration = time.time() - api_call_start_timestamp
                
                # Extract token count info (from extracted_response or api_response)
                input_tokens = output_tokens = total_tokens = 0
                if extracted_response and extracted_response.get("metadata", {}).get("usage"):
                    usage = extracted_response["metadata"]["usage"]
                    input_tokens = usage.get("prompt_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                elif api_response is not None:
                    # Even if extracted_response is None, try to extract from api_response
                    try:
                        if hasattr(api_response, 'usage') and api_response.usage:
                            input_tokens = api_response.usage.prompt_tokens if hasattr(api_response.usage, 'prompt_tokens') else 0
                            output_tokens = api_response.usage.completion_tokens if hasattr(api_response.usage, 'completion_tokens') else 0
                            total_tokens = api_response.usage.total_tokens if hasattr(api_response.usage, 'total_tokens') else (input_tokens + output_tokens)
                    except Exception:
                        pass
                
                self._record_api_usage(
                    messages, 
                    extracted_response, 
                    success=success, 
                    error_message=error_message,
                    api_call_start_time=api_call_start_time_str,
                    api_call_duration=api_call_duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    **kwargs
                )
        
        # If we reach here, all retries have been exhausted
        newline = '\n'
        filtered_errors = [msg for msg in error_messages if msg and msg.strip()]
        if not filtered_errors:
            filtered_errors = ["Unknown error occurred during LLM generation"]
        error_content = f"Failed to generate response after {max_attempts} attempts:\n{newline.join(filtered_errors)}"
        raise LLMError(error_content)
    
    def _record_api_usage(
        self,
        messages: List[Dict[str, Any]],
        response: Optional[Dict[str, Any]],
        success: bool,
        error_message: str = "",
        api_call_start_time: Optional[str] = None,
        api_call_duration: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        **kwargs
    ) -> None:
        """
        Record API usage (internal method).
        
        Args:
            messages: Input message list
            response: API response (on success) or None (on failure)
            success: Whether the call succeeded
            error_message: Error message (on failure)
            api_call_start_time: API call start time (format: yyyy-mm-dd_hh-mm-ss)
            api_call_duration: API call duration in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            total_tokens: Total number of tokens
            **kwargs: Additional parameters
        """
        if self.api_usage_recorder is None:
            return
        
        try:
            # Add call timing and token info to kwargs for _create_usage_record
            record_kwargs = {**kwargs}
            if api_call_start_time is not None:
                record_kwargs['api_call_start_time'] = api_call_start_time
            if api_call_duration is not None:
                record_kwargs['api_call_duration'] = api_call_duration
            
            record = self._create_usage_record(
                messages, response, success, error_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                **record_kwargs
            )
            self.api_usage_recorder.record(record)
        except Exception:
            # Recording failure should not affect the main program
            pass
