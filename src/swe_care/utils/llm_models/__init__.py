from typing import Any

from loguru import logger

from swe_care.utils.llm_models.clients import (
    AnthropicClient,
    BaseModelClient,
    DeepSeekClient,
    GeminiClient,
    MoonshotClient,
    OpenAIClient,
    QwenClient,
)

# Map of available LLM clients cited from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
LLM_CLIENT_MAP = {
    "openai": {
        "client_class": OpenAIClient,
        "models": [
            {"name": "gpt-4o", "max_input_tokens": 128000},
            {"name": "gpt-4o-mini", "max_input_tokens": 128000},
            {"name": "gpt-4.1", "max_input_tokens": 1047576},
            {"name": "gpt-4.5-preview", "max_input_tokens": 128000},
            {"name": "gpt-5", "max_input_tokens": 128000},
            {"name": "gpt-5-chat", "max_input_tokens": 128000},
            {"name": "o1", "max_input_tokens": 200000},
            {"name": "o1-mini", "max_input_tokens": 128000},
            {"name": "o3", "max_input_tokens": 200000},
            {"name": "o3-mini", "max_input_tokens": 200000},
        ],
    },
    "anthropic": {
        "client_class": AnthropicClient,
        "models": [
            {"name": "claude-opus-4", "max_input_tokens": 200000},
            {"name": "claude-sonnet-4", "max_input_tokens": 200000},
            {"name": "claude-3-7-sonnet", "max_input_tokens": 200000},
        ],
    },
    "deepseek": {
        "client_class": DeepSeekClient,
        "models": [
            {"name": "deepseek-chat", "max_input_tokens": 128000},  # DeepSeek-V3.1
            {"name": "DeepSeek-V3.1", "max_input_tokens": 128000},
            {
                "name": "deepseek-reasoner",
                "max_input_tokens": 128000,
            },  # DeepSeek-V3.1 thinking
        ],
    },
    "qwen": {
        "client_class": QwenClient,
        "models": [
            {"name": "qwen3-32b", "max_input_tokens": 128000},
            {"name": "qwen3-30b-a3b", "max_input_tokens": 128000},
            {"name": "qwen3-235b-a22b", "max_input_tokens": 128000},
        ],
    },
    "moonshot": {
        "client_class": MoonshotClient,
        "models": [
            {"name": "kimi-k2-0711-preview", "max_input_tokens": 131072},
            {"name": "kimi-k2-0905-preview", "max_input_tokens": 131072},
        ],
    },
    "gemini": {
        "client_class": GeminiClient,
        "models": [
            {"name": "gemini-2.5-pro", "max_input_tokens": 1048576},
        ],
    },
}


def get_available_models_and_providers() -> tuple[list[str], list[str]]:
    """Get available models and providers from LLM_CLIENT_MAP."""
    available_providers = list(LLM_CLIENT_MAP.keys())
    available_models = []
    for provider_info in LLM_CLIENT_MAP.values():
        available_models.extend([model["name"] for model in provider_info["models"]])
    return available_providers, available_models


def init_llm_client(
    model: str, model_provider: str, **model_kwargs: Any
) -> BaseModelClient:
    """Initialize an LLM client.

    Args:
        model: Model name
        model_provider: Provider name (openai, anthropic)
        **model_kwargs: Additional model arguments

    Returns:
        Initialized LLM client

    Raises:
        ValueError: If the model provider or model is not supported
    """
    if model_provider not in LLM_CLIENT_MAP:
        raise ValueError(
            f"Unsupported model provider: {model_provider}. "
            f"Supported providers: {list(LLM_CLIENT_MAP.keys())}"
        )

    provider_info = LLM_CLIENT_MAP[model_provider]

    _, model_list = get_available_models_and_providers()

    # if model not in model_list:
    #     logger.warning(
    #         f"Model {model} not in known models for {model_provider}. "
    #         f"Known models: {provider_info['models']}. Proceeding anyway..."
    #     )

    client_class = provider_info["client_class"]
    return client_class(model, model_provider, **model_kwargs)


def parse_model_args(model_args_str: str | None) -> dict[str, Any]:
    """Parse model arguments string into a dictionary.

    Args:
        model_args_str: Comma-separated string of key=value pairs

    Returns:
        Dictionary of parsed arguments

    Example:
        "top_p=0.95,temperature=0.70" -> {"top_p": 0.95, "temperature": 0.70}
    """
    if not model_args_str:
        return {}

    args = {}
    for pair in model_args_str.split(","):
        if "=" not in pair:
            logger.warning(f"Skipping invalid model argument: {pair}")
            continue

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        try:
            # Try int first
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                args[key] = int(value)
            # Try float
            elif "." in value and value.replace(".", "").replace("-", "").isdigit():
                args[key] = float(value)
            # Try boolean
            elif value.lower() in ("true", "false"):
                args[key] = value.lower() == "true"
            # Keep as string
            else:
                args[key] = value
        except ValueError:
            args[key] = value

    return args


def get_model_info(model_name: str) -> tuple[str, int]:
    """Get normalized model name and max input tokens for a given model.

    Args:
        model_name: Model name that might be from different providers
                    (e.g., 'us.anthropic.claude-sonnet-4-20250514-v1:0')

    Returns:
        Tuple of (normalized_model_name, max_input_tokens)

    Raises:
        ValueError: If model cannot be found
    """
    # Normalize model name by checking for known patterns
    normalized_name = None
    max_tokens = None

    # Check each provider's models
    for provider_info in LLM_CLIENT_MAP.values():
        for model_info in provider_info["models"]:
            model_canonical_name = model_info["name"]

            # Check if the canonical name is contained in the input model name
            if model_canonical_name in model_name:
                normalized_name = model_canonical_name
                max_tokens = model_info["max_input_tokens"]
                break

            # Also check for partial matches (e.g., "claude-sonnet-4" in "claude-sonnet-4-20250514")
            if "-" in model_canonical_name:
                parts = model_canonical_name.split("-")
                # Try different combinations of parts
                for i in range(len(parts), 0, -1):
                    partial = "-".join(parts[:i])
                    if (
                        partial in model_name and len(partial) > 3
                    ):  # Avoid too short matches
                        normalized_name = model_canonical_name
                        max_tokens = model_info["max_input_tokens"]
                        break

        if normalized_name:
            break

    if not normalized_name:
        # If no exact match found, log warning and return defaults
        logger.warning(
            f"Could not find exact match for model '{model_name}' in LLM_CLIENT_MAP. "
            "Using default token limit."
        )
        # Return the original model name with a conservative default
        return model_name, 128000  # Conservative default

    return normalized_name, max_tokens
