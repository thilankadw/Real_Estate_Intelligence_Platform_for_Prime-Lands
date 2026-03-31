"""
Application layer - services and use cases.

Contains:
- chat_service: RAG, CAG, CRAG services
- ingest_documents_service: Crawling, chunking, indexing
- evaluation_service: Metrics and evaluation
"""

from .ingest_document_service import (
    ChunkingService,
    PrimeLandsWebCrawler
)


__all__ = [
    # Ingest services
    "ChunkingService",
    "PrimeLandsWebCrawler",
    # Chat services
    # "RAGService",
    # "CAGService",
    # "CRAGService",
    # "CAGCache",
    # "build_rag_chain"
]
