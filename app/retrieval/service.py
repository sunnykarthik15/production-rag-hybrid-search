"""
Dense Retrieval Service.

Integrates EmbeddingService and VectorStore to execute similarity searches
and return structured RetrievalResponse objects.
"""

from __future__ import annotations

import logging

from app.config import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K
from app.embeddings.service import EmbeddingService
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.vector_store import IndexNotFoundError, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetrievalError(Exception):
    """Base exception for retrieval operation failures."""


# ---------------------------------------------------------------------------
# DenseRetrievalService Implementation
# ---------------------------------------------------------------------------


class DenseRetrievalService:
    """Service providing end-to-end dense vector retrieval.

    Embeds incoming query strings using EmbeddingService and queries
    the VectorStore to return structured RetrievalResponse objects.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()

    def initialize(self) -> None:
        """Load vector store index from disk.

        Raises
        ------
        IndexNotFoundError
            If index files do not exist.
        RetrievalError
            If loading fails.
        """
        try:
            self.vector_store.load()
        except IndexNotFoundError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Failed to initialize vector store: {exc}") from exc

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResponse:
        """Retrieve top_k nearest chunk results for a query string.

        Parameters
        ----------
        query : str
            Input natural language query text.
        top_k : int
            Number of top results to retrieve (between 1 and 150).

        Returns
        -------
        RetrievalResponse
            Container with query, top_k, and ordered RetrievalResult list.

        Raises
        ------
        RetrievalError
            If query is empty, top_k is invalid, or vector store is uninitialized.
        """
        if not query or not query.strip():
            raise RetrievalError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RetrievalError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        if not self.vector_store.is_initialized:
            # Try auto-loading from default path if not yet initialized
            try:
                self.initialize()
            except IndexNotFoundError as exc:
                raise RetrievalError(
                    "Vector store is not initialized and no saved index was found on disk. "
                    "Run scripts/build_vector_index.py first."
                ) from exc

        # 1. Embed query
        try:
            query_vector = self.embedding_service.embed_text(query)
        except Exception as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc

        # 2. Vector search
        try:
            raw_results = self.vector_store.search(query_vector, top_k=top_k)
        except VectorStoreError as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        # 3. Format structured results
        formatted_results = [
            RetrievalResult(
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                text=meta["text"],
                title=meta["title"],
                department=meta["department"],
                similarity_score=score,
                rank=rank,
            )
            for meta, score, rank in raw_results
        ]

        return RetrievalResponse(
            query=query.strip(),
            results=formatted_results,
            top_k=top_k,
        )
