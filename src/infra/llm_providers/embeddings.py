"""Factories for embedding clients configured through project settings.

The helpers in this module wrap :class:`langchain_openai.OpenAIEmbeddings`
and support two deployment modes used by the project:

- direct OpenAI access
- OpenRouter's OpenAI-compatible embeddings endpoint
"""

from typing import Optional, Any
import os
from langchain_openai import OpenAIEmbeddings

from config import (
    PROVIDER,
    OPENROUTER_BASE_URL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_SHOW_PROGRESS,
    get_api_key,
    get_embedding_model,
)


def get_default_embeddings(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    tier: str = "default",
    batch_size: Optional[int] = None,
    show_progress: Optional[bool] = None,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """
    Build the default embedding client for the configured environment.

    Args:
        model: Explicit embedding model name. When omitted, the model is
            resolved from :mod:`config` using ``provider`` and ``tier``.

        provider: Provider key to resolve credentials and endpoint settings.
            This implementation supports ``"openai"`` and ``"openrouter"``.

        tier: Logical embedding tier defined in ``config/models.yaml``.

        batch_size: Batch size forwarded to LangChain as ``chunk_size``.

        show_progress: Whether LangChain should display a progress bar.
        
        **kwargs: Additional keyword arguments forwarded to
            :class:`langchain_openai.OpenAIEmbeddings`.

    Returns:
        A configured :class:`langchain_openai.OpenAIEmbeddings` instance.

    Notes:
        If ``model`` includes a provider prefix such as
        ``"openai/text-embedding-3-large"``, only the trailing model id is
        passed to LangChain because the embeddings client expects the bare
        model name.

    Examples:
        >>> embedder = get_default_embeddings()
        >>> embedder = get_default_embeddings(tier="small")
        >>> embedder = get_default_embeddings(model="text-embedding-3-small")
    """
    # Determine provider
    use_provider = provider or PROVIDER
    
    # Determine model
    if model:
        use_model = model
    else:
        use_model = get_embedding_model(provider=use_provider, tier=tier)
    
    # Strip provider prefix if present (embeddings use bare model name)
    if "/" in use_model:
        use_model = use_model.split("/")[-1]
    
    # Get API key
    api_key = get_api_key(use_provider)
    if not api_key:
        # Fallback to OPENAI_API_KEY for backward compatibility
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Set defaults
    use_batch_size = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE
    use_show_progress = show_progress if show_progress is not None else EMBEDDING_SHOW_PROGRESS
    
    # Configure based on provider
    if use_provider == "openrouter":
        # OpenRouter embeddings (uses OpenAI-compatible API)
        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            chunk_size=use_batch_size,
            show_progress_bar=use_show_progress,
            **kwargs
        )
    else:
        # Direct OpenAI access
        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            chunk_size=use_batch_size,
            show_progress_bar=use_show_progress,
            **kwargs
        )


def get_small_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """Return the project's lower-cost embedding configuration."""
    return get_default_embeddings(tier="small", **kwargs)
