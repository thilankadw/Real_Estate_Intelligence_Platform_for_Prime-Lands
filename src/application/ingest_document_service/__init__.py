"""
Document ingestion service for Prime Lands source content.

Provides:
- Web crawling for Prime Lands website content extraction
- Public package exports for ingestion services
"""

from .web_crawler import PrimeLandsWebCrawler
from .chunkers import (
    ChunkingService,
    semantic_chunk,
    fixed_chunk,
    sliding_chunk,
    parent_child_chunk,
    late_chunk_index,
    late_chunk_split,
    count_tokens
)

__all__ = [
    # Chunking
    "ChunkingService",
    "semantic_chunk",
    "fixed_chunk",
    "sliding_chunk",
    "parent_child_chunk",
    "late_chunk_index",
    "late_chunk_split",
    "count_tokens",
    #Crawling
    "PrimeLandsWebCrawler",
]
