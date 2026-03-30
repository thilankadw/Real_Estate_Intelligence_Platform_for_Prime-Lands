"""Public factory interface for LLM and embedding providers.

Import from this package when application code needs a configured chat model or
embedding client without depending on provider-specific setup details.
"""

from .llm_services import (
    get_chat_llm,
    get_reasoning_llm,
    get_strong_llm,
    list_available_models,
)
from .embeddings import (
    get_default_embeddings,
    get_small_embeddings,
)

__all__ = [
    # Chat LLMs
    "get_chat_llm",
    "get_reasoning_llm",
    "get_strong_llm",
    "list_available_models",
    # Embeddings
    "get_default_embeddings",
    "get_small_embeddings",
]
