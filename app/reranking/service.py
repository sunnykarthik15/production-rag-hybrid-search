"""Cross-Encoder Reranking Service.

Provides fine-grained second-stage relevance scoring for retrieved candidate chunks
using SentenceTransformers CrossEncoder models.
"""

from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder

from app.config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    RERANKER_MODEL_NAME,
)
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import RetrievalError

logger = logging.getLogger(__name__)


class RerankerError(RetrievalError):
    """Exception raised for failures during reranking operations."""


class RerankerService:
    """Service providing Cross-Encoder reranking for candidate retrieval results."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        """Initialize RerankerService.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face cross-encoder model name. Defaults to RERANKER_MODEL_NAME.
        """
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def is_initialized(self) -> bool:
        """Return True if the CrossEncoder model is loaded and ready."""
        return self._model is not None

    def initialize(self) -> None:
        """Load the CrossEncoder model into memory.

        Raises
        ------
        RerankerError
            If model loading fails.
        """
        if self._model is not None:
            return

        try:
            logger.info(f"Loading CrossEncoder model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder model loaded successfully.")
        except Exception as exc:
            raise RerankerError(f"Failed to load CrossEncoder model '{self.model_name}': {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResponse:
        """Rerank input candidate results for a query string using the CrossEncoder model.

        Parameters
        ----------
        query : str
            Input natural language query text.
        candidates : List[RetrievalResult]
            List of candidate retrieval results from first-stage retrieval.
        top_k : int
            Number of top reranked results to return.

        Returns
        -------
        RetrievalResponse
            Structured container with query, top_k, and ordered RetrievalResult list.

        Raises
        ------
        RerankerError
            If query is empty, top_k is invalid, or scoring fails.
        """
        if not query or not query.strip():
            raise RerankerError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RerankerError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        if not candidates:
            return RetrievalResponse(
                query=query.strip(),
                results=[],
                top_k=top_k,
            )

        if not self.is_initialized or self._model is None:
            try:
                self.initialize()
            except Exception as exc:
                raise RerankerError(f"Auto-initialization of RerankerService failed: {exc}") from exc

        try:
            # Pair query with candidate text (including title context)
            pairs = [(query, f"{c.title}: {c.text}") for c in candidates]
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankerError(f"CrossEncoder scoring failed: {exc}") from exc

        # Pair candidate items with their cross-encoder score
        scored_candidates = [
            (c, float(score)) for c, score in zip(candidates, scores)
        ]

        # Sort primarily by score descending, tie-break by chunk_id ascending
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda item: (-item[1], item[0].chunk_id),
        )

        # Build updated top_k RetrievalResult objects
        reranked_results: List[RetrievalResult] = []
        for rank, (item, score) in enumerate(sorted_candidates[:top_k], start=1):
            reranked_results.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    text=item.text,
                    title=item.title,
                    department=item.department,
                    similarity_score=score,
                    rank=rank,
                )
            )

        return RetrievalResponse(
            query=query.strip(),
            results=reranked_results,
            top_k=top_k,
        )
