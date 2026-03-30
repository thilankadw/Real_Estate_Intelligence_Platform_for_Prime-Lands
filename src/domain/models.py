"""
Core domain models for the Prime Lands intelligence platform.

Defines the document entities shared across ingestion and downstream processing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class Document:
    """
    Represents a normalized document captured by the ingestion pipeline.

    Attributes:
        url: Source URL of the document
        title: Document or page title
        content: Full text content in Markdown format
        metadata: Additional metadata such as headings, links, or crawl depth
    """
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate required document fields."""
        if not self.url:
            raise ValueError("Document URL cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
