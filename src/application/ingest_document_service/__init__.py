"""
Document ingestion service for Prime Lands source content.

Provides:
- Web crawling for Prime Lands website content extraction
- Public package exports for ingestion services
"""

from .web_crawler import PrimeLandsWebCrawler

__all__ = [
    "PrimeLandsWebCrawler",
]
