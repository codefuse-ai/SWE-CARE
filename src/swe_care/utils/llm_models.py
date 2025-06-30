import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

try:
    import openai
except ImportError:
    raise ImportError(
        "OpenAI package not found. Please install it with: pip install openai"
    )

try:
    import anthropic
except ImportError:
    raise ImportError(
        "Anthropic package not found. Please install it with: pip install anthropic"
    )

DEFAULT_MAX_RETRIES = 4


class BaseModelClient(ABC):
    """Abstract base class for LLM model clients."""

    client: Any
    model: str
    model_provider: str
    model_kwargs: dict[str, Any]
    max_retries: int

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        self.model = model
        self.model_provider = model_provider
        self.model_kwargs = model_kwargs
        self.max_retries = max_retries

    @abstractmethod
    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using the LLM API.

        Args:
            messages: List of messages in OpenAI format [{"role": "user", "content": "..."}]

        Returns:
            The generated completion text
        """
        pass


class OpenAIClient(BaseModelClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        # Initialize the OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key, max_retries=self.max_retries)

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.model_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error creating OpenAI completion: {e}")
            raise e


class DeepSeekClient(OpenAIClient):
    """DeepSeek API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        self.client = openai.OpenAI.copy(
            self.client, base_url="https://api.deepseek.com/v1"
        )


class QwenClient(OpenAIClient):
    """Qwen API client."""

    def __init__(self, model: str, model_provider: str, **model_kwargs: Any):
        # Handle enable_thinking
        if "enable_thinking" in model_kwargs:
            enable_thinking = model_kwargs.pop("enable_thinking")
            model_kwargs["extra_body"] = {"enable_thinking": enable_thinking}
        else:
            model_kwargs["extra_body"] = {"enable_thinking": False}

        super().__init__(model, model_provider, **model_kwargs)

        self.client = openai.OpenAI.copy(
            self.client, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


class AnthropicClient(BaseModelClient):
    """Anthropic API client."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **model_kwargs: Any,
    ):
        super().__init__(model, model_provider, max_retries, **model_kwargs)

        # Initialize the Anthropic client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=self.max_retries)

    def create_completion(self, messages: list[dict[str, str]]) -> str:
        """Create a completion using Anthropic API."""
        try:
            # Convert OpenAI format to Anthropic format
            if messages and messages[0]["role"] == "system":
                system_message = messages[0]["content"]
                messages = messages[1:]
            else:
                system_message = None

            # Anthropic expects alternating user/assistant messages
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            kwargs = self.model_kwargs.copy()
            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                max_tokens=kwargs.pop("max_tokens", 4096),
                **kwargs,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error creating Anthropic completion: {e}")
            raise e


# Map of available LLM clients
LLM_CLIENT_MAP = {
    "openai": {
        "client_class": OpenAIClient,
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.5-preview",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
        ],
    },
    "anthropic": {
        "client_class": AnthropicClient,
        "models": [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
        ],
    },
    "deepseek": {
        "client_class": DeepSeekClient,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "qwen": {
        "client_class": QwenClient,
        "models": ["qwen3-32b", "qwen3-30b-a3b", "qwen3-235b-a22b"],
    },
}


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

    if model not in provider_info["models"]:
        logger.warning(
            f"Model {model} not in known models for {model_provider}. "
            f"Known models: {provider_info['models']}. Proceeding anyway..."
        )

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
