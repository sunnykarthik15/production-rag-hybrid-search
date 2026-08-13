"""Retrieval package for vector and keyword search."""

from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.bm25_store import BM25Store, BM25StoreError, tokenize
from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import DenseRetrievalService, RetrievalError
from app.retrieval.vector_store import (
    IndexNotFoundError,
    MetadataLoadError,
    VectorStore,
    VectorStoreError,
)

__all__ = [
    # Services & Core Models
    "DenseRetrievalService",
    "BM25RetrievalService",
    "HybridRetrievalService",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalError",
    # Dense Vector Store
    "VectorStore",
    "VectorStoreError",
    "IndexNotFoundError",
    "MetadataLoadError",
    # Lexical BM25 Store
    "BM25Store",
    "BM25StoreError",
    "tokenize",
]
