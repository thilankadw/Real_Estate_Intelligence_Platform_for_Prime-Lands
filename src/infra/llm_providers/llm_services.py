"""Factories for chat LLM clients used by the infrastructure layer.

The module standardizes project access to chat models through
``langchain_openai.ChatOpenAI``. It supports:

- direct OpenAI access
- OpenRouter's OpenAI-compatible chat completions API

Provider-specific model ids can still be selected through OpenRouter by using
the corresponding entries from ``config/models.yaml``.
"""

from typing import Optional, Any
import os
from langchain_openai import ChatOpenAI

from config import (
    PROVIDER,
    CHAT_MODEL,
    OPENROUTER_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_STREAMING,
    get_api_key,
    get_chat_model,
)


def get_chat_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tier: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
    **kwargs: Any
) -> ChatOpenAI:
    """
    Build a chat LLM client from configuration and optional overrides.

    Args:
        model: Explicit model identifier. When omitted, the model is resolved
            from ``config/models.yaml`` using ``provider`` and ``tier``.

        provider: Provider key used to resolve credentials and endpoint
            settings. This implementation is intended for ``"openai"`` and
            ``"openrouter"``.

        tier: Logical model tier from the project model catalog.

        temperature: Sampling temperature passed to ``ChatOpenAI``.

        max_tokens: Maximum completion length passed to ``ChatOpenAI``.

        streaming: Whether token streaming should be enabled.
        
        **kwargs: Additional keyword arguments forwarded to
            :class:`langchain_openai.ChatOpenAI`.

    Returns:
        A configured :class:`langchain_openai.ChatOpenAI` instance.

    Examples:
        >>> llm = get_chat_llm()
        >>> llm = get_chat_llm(model="anthropic/claude-3-5-sonnet")
        >>> llm = get_chat_llm(tier="strong")
        >>> llm = get_chat_llm(provider="openai")
    """
    # Determine provider
    use_provider = provider or PROVIDER
    
    # Determine model
    if model:
        use_model = model
    elif tier:
        use_model = get_chat_model(provider=use_provider, tier=tier)
    else:
        use_model = CHAT_MODEL
    
    # Get API key
    api_key = get_api_key(use_provider)
    if not api_key:
        # Fallback to OPENAI_API_KEY for backward compatibility
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Set defaults
    use_temperature = temperature if temperature is not None else LLM_TEMPERATURE
    use_max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    use_streaming = streaming if streaming is not None else LLM_STREAMING
    
    # Configure based on provider
    if use_provider == "openrouter":
        # OpenRouter configuration
        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/thilankadw/Real_Estate_Intelligence_Platform_for_Prime-Lands",
                "X-Title": "Prime Lands Intelligence Platform"
            },
            **kwargs
        )
    else:
        # Direct provider access (OpenAI compatible)
        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            **kwargs
        )


def get_reasoning_llm(**kwargs: Any) -> ChatOpenAI:
    """Return the chat model configured for reasoning-heavy workloads."""
    return get_chat_llm(tier="reason", **kwargs)


def get_strong_llm(**kwargs: Any) -> ChatOpenAI:
    """Return the chat model configured for higher-capability workloads."""
    return get_chat_llm(tier="strong", **kwargs)


def list_available_models() -> dict:
    """Return the parsed model catalog from ``config/models.yaml``."""
    from config import get_all_models
    return get_all_models()
