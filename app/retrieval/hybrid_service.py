"""Hybrid Retrieval Service using Reciprocal Rank Fusion (RRF).

Combines dense vector retrieval and BM25 lexical retrieval candidate lists
using Reciprocal Rank Fusion (RRF) to output optimal fused rankings.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from app.config import (
    DEFAULT_HYBRID_CANDIDATE_K,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
)
from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import DenseRetrievalService, RetrievalError

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Service providing hybrid search via Reciprocal Rank Fusion (RRF).

    Combines candidate result lists from DenseRetrievalService and
    BM25RetrievalService using standard RRF rank scoring.
    """

    def __init__(
        self,
        dense_service: DenseRetrievalService | None = None,
        bm25_service: BM25RetrievalService | None = None,
    ) -> None:
        """Initialize HybridRetrievalService with sub-services.

        Parameters
        ----------
        dense_service : DenseRetrievalService, optional
            Dense vector retrieval service instance.
        bm25_service : BM25RetrievalService, optional
            BM25 lexical retrieval service instance.
        """
        self.dense_service = dense_service or DenseRetrievalService()
        self.bm25_service = bm25_service or BM25RetrievalService()

    def initialize(self) -> None:
        """Initialize underlying dense and BM25 retrieval stores.

        Raises
        ------
        RetrievalError
            If initializing either underlying service fails.
        """
        try:
            self.dense_service.initialize()
            self.bm25_service.initialize()
        except Exception as exc:
            raise RetrievalError(f"Failed to initialize hybrid retrieval service: {exc}") from exc

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_HYBRID_CANDIDATE_K,
    ) -> RetrievalResponse:
        """Retrieve top_k hybrid search results using Reciprocal Rank Fusion.

        Parameters
        ----------
        query : str
            Input natural language query text.
        top_k : int
            Number of final hybrid results to return (>= 1, <= 150).
        rrf_k : int
            RRF smoothing parameter (>= 1). Standard default is 60.
        candidate_k : int
            Number of candidate results to fetch from each underlying retriever (>= top_k).

        Returns
        -------
        RetrievalResponse
            Structured container with query, top_k, and ordered RetrievalResult list.

        Raises
        ------
        RetrievalError
            If query is empty/whitespace, parameters are out of range, or sub-services fail.
        """
        if not query or not query.strip():
            raise RetrievalError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RetrievalError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        if not isinstance(candidate_k, int) or candidate_k < top_k or candidate_k > MAX_TOP_K:
            raise RetrievalError(
                f"candidate_k must be an integer >= top_k ({top_k}) and <= {MAX_TOP_K}, got {candidate_k}."
            )

        if not isinstance(rrf_k, (int, float)) or rrf_k < 1:
            raise RetrievalError(f"rrf_k must be a number >= 1, got {rrf_k}.")

        # 1. Fetch candidate lists from Dense and BM25 retrievers
        try:
            dense_response = self.dense_service.retrieve(query, top_k=candidate_k)
        except Exception as exc:
            raise RetrievalError(f"Dense candidate retrieval failed: {exc}") from exc

        try:
            bm25_response = self.bm25_service.retrieve(query, top_k=candidate_k)
        except Exception as exc:
            raise RetrievalError(f"BM25 candidate retrieval failed: {exc}") from exc

        # 2. Reciprocal Rank Fusion (RRF) calculation
        scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, RetrievalResult] = {}

        for res in dense_response.results:
            scores[res.chunk_id] += 1.0 / (rrf_k + res.rank)
            chunk_map[res.chunk_id] = res

        for res in bm25_response.results:
            scores[res.chunk_id] += 1.0 / (rrf_k + res.rank)
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res

        if not scores:
            return RetrievalResponse(
                query=query.strip(),
                results=[],
                top_k=top_k,
            )

        # 3. Primary sort by RRF score descending, secondary tie-break by chunk_id ascending
        sorted_chunk_ids = sorted(
            scores.keys(),
            key=lambda cid: (-scores[cid], cid),
        )

        # 4. Construct final top_k RetrievalResult list with 1-based ranks
        fused_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            item = chunk_map[cid]
            fused_results.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    text=item.text,
                    title=item.title,
                    department=item.department,
                    similarity_score=scores[cid],
                    rank=rank,
                )
            )

        return RetrievalResponse(
            query=query.strip(),
            results=fused_results,
            top_k=top_k,
        )
