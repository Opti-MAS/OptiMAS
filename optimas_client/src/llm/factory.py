"""LLM factory"""
import yaml
from typing import Dict, Any, Optional, Type
from pathlib import Path
from .base import BaseLLM
from .openai_llm import OpenAILLM
from .gemini3_llm import GeminiLLM
from .usage_recorder import APIUsageRecorder
from ..utils.exceptions import ConfigError

# LLM class mapping
LLM_CLASS_MAP: Dict[str, Type[BaseLLM]] = {
    "OpenAILLM": OpenAILLM,
    "GeminiLLM": GeminiLLM,
    # Anthropic is not wired up in this release; add an AnthropicLLM
    # implementing BaseLLM and register it here to enable Claude backbones.
}

# Model configuration cache
_model_configs: Optional[Dict[str, Dict[str, Any]]] = None


def _load_model_configs() -> Dict[str, Dict[str, Any]]:
    """Load model configuration file."""
    global _model_configs
    if _model_configs is not None:
        return _model_configs
    
    # Locate the config file path
    config_path = Path(__file__).parent.parent.parent / "config" / "model_configs.yaml"
    if not config_path.exists():
        raise ConfigError(f"Model config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    if not isinstance(config_data, dict) or "models" not in config_data:
        raise ConfigError("Invalid model config file format: missing 'models' key")
    
    _model_configs = config_data["models"]
    return _model_configs


# Local model prefix: names with this prefix use LOCAL_LLM_*; stripped when requesting sglang
LOCAL_MODEL_PREFIX = "local_"


def get_llm_class_by_model_name(model_name: str) -> Type[BaseLLM]:
    """
    Get the corresponding LLM class by model name.
    If the name starts with local_, look up the class using the name with the prefix stripped
    (reusing existing entries in model_configs).
    """
    model_configs = _load_model_configs()
    lookup_name = model_name[len(LOCAL_MODEL_PREFIX):] if model_name.startswith(LOCAL_MODEL_PREFIX) else model_name

    if lookup_name not in model_configs:
        if model_name.startswith(LOCAL_MODEL_PREFIX):
            return OpenAILLM
        available_models = list(model_configs.keys())
        raise ConfigError(
            f"Model '{model_name}' not found in model configs. "
            f"Available models: {', '.join(available_models) if available_models else 'None'}"
        )
    
    model_config = model_configs[lookup_name]
    class_name = model_config.get("class")
    
    if not class_name:
        raise ConfigError(f"Model '{model_name}' config missing 'class' field")
    
    if class_name not in LLM_CLASS_MAP:
        available_classes = list(LLM_CLASS_MAP.keys())
        raise ConfigError(
            f"LLM class '{class_name}' not found in LLM_CLASS_MAP. "
            f"Available classes: {', '.join(available_classes) if available_classes else 'None'}"
        )
    
    return LLM_CLASS_MAP[class_name]


def create_llm(
    config: Any,
    recording_config: Optional[Dict[str, Any]] = None,
    global_llm_settings: Optional[Dict[str, Any]] = None
) -> BaseLLM:
    """
    Create an LLM instance from configuration.
    
    Args:
        config: LLM config dict; must contain 'model_name', optionally 'max_retries'
        recording_config: API usage recording config (optional); contains 'enabled' and 'storage_path'.
                        If None or enabled=False, no recorder is created.
        global_llm_settings: Global LLM settings (optional) such as 'timeout'; merged into config in BaseLLM
    
    Returns:
        LLM instance
    
    Raises:
        ConfigError: If the config is invalid or the model is unsupported
    """
    if not isinstance(config, dict):
        raise ConfigError("LLM config must be a dictionary")
    
    model_name = config.get("model_name")
    if not model_name:
        raise ConfigError("LLM config must contain 'model_name' field")
    
    # Get the LLM class by model name
    llm_class = get_llm_class_by_model_name(model_name)
    
    api_usage_recorder = None
    if recording_config and recording_config.get("enabled", False):
        api_usage_recorder = APIUsageRecorder(
            storage_path=recording_config.get("storage_path"), model_name=model_name
        )
    
    # Pass through all config fields except model_name to the LLM class
    llm_config = {k: v for k, v in config.items() if k != "model_name"}
    
    # Local model port/URL is provided by config; parsed in OpenAILLM
    # Global settings are merged into config in BaseLLM.__init__
    
    return llm_class(
        model_name=model_name,
        api_usage_recorder=api_usage_recorder,
        config=llm_config,
        global_llm_settings=global_llm_settings
    )
