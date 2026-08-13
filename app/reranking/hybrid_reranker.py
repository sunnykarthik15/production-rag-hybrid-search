"""Reranked Hybrid Retrieval Service.

Integrates HybridRetrievalService (RRF) candidate retrieval with RerankerService
(Cross-Encoder) to output optimal second-stage reranked search results.
"""

from __future__ import annotations

import logging

from app.config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
)
from app.reranking.service import RerankerError, RerankerService
from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.models import RetrievalResponse
from app.retrieval.service import RetrievalError

logger = logging.getLogger(__name__)


class RerankedHybridService:
    """Service providing end-to-end Hybrid Search + Cross-Encoder Reranking."""

    def __init__(
        self,
        hybrid_service: HybridRetrievalService | None = None,
        reranker_service: RerankerService | None = None,
    ) -> None:
        """Initialize RerankedHybridService with sub-services.

        Parameters
        ----------
        hybrid_service : HybridRetrievalService, optional
            First-stage hybrid retrieval service instance.
        reranker_service : RerankerService, optional
            Second-stage cross-encoder reranking service instance.
        """
        self.hybrid_service = hybrid_service or HybridRetrievalService()
        self.reranker_service = reranker_service or RerankerService()

    def initialize(self) -> None:
        """Initialize underlying hybrid retrieval and reranker services.

        Raises
        ------
        RerankerError
            If initializing either underlying service fails.
        """
        try:
            self.hybrid_service.initialize()
            self.reranker_service.initialize()
        except Exception as exc:
            raise RerankerError(f"Failed to initialize reranked hybrid service: {exc}") from exc

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
    ) -> RetrievalResponse:
        """Execute Hybrid RRF candidate retrieval followed by Cross-Encoder reranking.

        Parameters
        ----------
        query : str
            Input natural language query text.
        top_k : int
            Number of final reranked results to return (>= 1, <= 150).
        rrf_k : int
            RRF smoothing parameter for first-stage hybrid candidate retrieval.
        candidate_k : int
            Number of hybrid candidates to retrieve before cross-encoder reranking (>= top_k).

        Returns
        -------
        RetrievalResponse
            Structured container with query, top_k, and ordered RetrievalResult list.

        Raises
        ------
        RerankerError
            If query is empty, parameters are invalid, or sub-services fail.
        """
        if not query or not query.strip():
            raise RerankerError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RerankerError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        if not isinstance(candidate_k, int) or candidate_k < top_k or candidate_k > MAX_TOP_K:
            raise RerankerError(
                f"candidate_k must be an integer >= top_k ({top_k}) and <= {MAX_TOP_K}, got {candidate_k}."
            )

        if not isinstance(rrf_k, (int, float)) or rrf_k < 1:
            raise RerankerError(f"rrf_k must be a number >= 1, got {rrf_k}.")

        # 1. First-stage hybrid candidate retrieval
        try:
            hybrid_response = self.hybrid_service.retrieve(
                query=query,
                top_k=candidate_k,
                rrf_k=rrf_k,
                candidate_k=max(candidate_k, DEFAULT_HYBRID_CANDIDATE_K),
            )
        except Exception as exc:
            raise RerankerError(f"First-stage hybrid retrieval failed: {exc}") from exc

        # 2. Second-stage Cross-Encoder reranking
        try:
            reranked_response = self.reranker_service.rerank(
                query=query,
                candidates=hybrid_response.results,
                top_k=top_k,
            )
        except Exception as exc:
            raise RerankerError(f"Second-stage reranking failed: {exc}") from exc

        return reranked_response
