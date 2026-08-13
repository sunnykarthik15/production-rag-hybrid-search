"""RAG Orchestration Service.

Connects first-stage Hybrid Retrieval (FAISS + BM25 via RRF), second-stage Cross-Encoder Reranking,
grounded context building, and LLM answer generation into a unified end-to-end RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    RAG_MIN_RELEVANCE_SCORE,
)
from app.generation.prompts import format_context_block
from app.generation.service import (
    INSUFFICIENT_INFO_ANSWER,
    GenerationError,
    GenerationService,
)
from app.rag.models import RAGResponse, RAGSource
from app.reranking.hybrid_reranker import RerankedHybridService
from app.reranking.service import RerankerError

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Exception raised for failures during end-to-end RAG orchestration."""


class RAGService:
    """Orchestrates Reranked Hybrid Search and grounded LLM Generation."""

    def __init__(
        self,
        hybrid_reranker: Optional[RerankedHybridService] = None,
        generation_service: Optional[GenerationService] = None,
        min_relevance_score: Optional[float] = None,
    ) -> None:
        """Initialize RAGService container.

        Parameters
        ----------
        hybrid_reranker : RerankedHybridService, optional
            Service executing hybrid retrieval and cross-encoder reranking.
        generation_service : GenerationService, optional
            Service executing grounded prompt generation.
        min_relevance_score : float, optional
            Minimum Cross-Encoder relevance score required to consider evidence sufficient.
            Defaults to configured RAG_MIN_RELEVANCE_SCORE.
        """
        self.hybrid_reranker = hybrid_reranker or RerankedHybridService()
        self.generation_service = generation_service or GenerationService()
        self.min_relevance_score = (
            min_relevance_score
            if min_relevance_score is not None
            else RAG_MIN_RELEVANCE_SCORE
        )

    def initialize(self) -> None:
        """Initialize underlying vector store, BM25 index, and reranker model.

        Raises
        ------
        RAGError
            If initializing underlying retrieval components fails.
        """
        try:
            self.hybrid_reranker.initialize()
        except Exception as exc:
            raise RAGError(f"Failed to initialize RAG retrieval components: {exc}") from exc

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        """Execute complete RAG pipeline: Query -> Reranked Retrieval -> Prompt -> Answer + Sources.

        Parameters
        ----------
        query : str
            Input natural language user question.
        top_k : int
            Number of reranked evidence chunks to pass to generation.

        Returns
        -------
        RAGResponse
            Structured response containing query, grounded answer text, and evidence citations.

        Raises
        ------
        RAGError
            If query is empty, top_k is invalid, or retrieval/generation sub-services fail.
        """
        if not query or not query.strip():
            raise RAGError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RAGError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        clean_query = query.strip()

        # 1. Execute Reranked Hybrid Retrieval
        try:
            retrieval_response = self.hybrid_reranker.retrieve(clean_query, top_k=top_k)
        except (RerankerError, Exception) as exc:
            raise RAGError(f"Retrieval step failed during RAG orchestration: {exc}") from exc

        # 2. Evidence Sufficiency Evaluation
        # Empty retrieval or top relevance score below min_relevance_score threshold
        if not retrieval_response.results:
            return RAGResponse(
                query=clean_query,
                answer=INSUFFICIENT_INFO_ANSWER,
                sources=[],
            )

        top_score = retrieval_response.results[0].similarity_score
        if top_score < self.min_relevance_score:
            logger.info(
                f"Top evidence score ({top_score:.4f}) below threshold ({self.min_relevance_score:.4f}). "
                "Returning grounded insufficient information response without invoking LLM."
            )
            return RAGResponse(
                query=clean_query,
                answer=INSUFFICIENT_INFO_ANSWER,
                sources=[],
            )

        # Map RetrievalResult items to RAGSource citation objects
        sources: List[RAGSource] = [
            RAGSource(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                title=item.title,
                department=item.department,
                rank=item.rank,
                score=item.similarity_score,
            )
            for item in retrieval_response.results
        ]

        # 3. Format grounded context block
        formatted_context = format_context_block(retrieval_response.results)

        # 4. Execute LLM Generation
        try:
            gen_response = self.generation_service.generate(clean_query, formatted_context)
        except (GenerationError, Exception) as exc:
            raise RAGError(f"Generation step failed during RAG orchestration: {exc}") from exc

        return RAGResponse(
            query=clean_query,
            answer=gen_response.answer,
            sources=sources,
        )
