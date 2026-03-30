"""
Domain layer - core business models.

Contains:
- models: Domain data models
- Document: Core document entity used across ingestion and processing
"""

from .models import Document

__all__ = [
    "Document",
]
