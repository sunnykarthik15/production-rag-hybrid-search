"""Integration tests for Cross-Encoder Reranker and Reranked Hybrid Service.

Runs actual inference with cross-encoder model against the real 150-chunk corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import FAISS_INDEX_PATH, METADATA_PATH
from app.embeddings.service import EmbeddingService
from app.reranking.hybrid_reranker import RerankedHybridService
from app.reranking.service import RerankerService
from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.models import RetrievalResponse
from app.retrieval.service import DenseRetrievalService
from app.retrieval.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"
_DATASET_PRESENT = _CHUNKS_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _DATASET_PRESENT,
    reason="Chunks dataset not found. Skipping reranker integration tests.",
)


class TestRerankerIntegration:
    """Integration test suite for Cross-Encoder Reranking against real corpus."""

    @pytest.fixture(scope="class")
    def reranked_service(self) -> RerankedHybridService:
        """Initialize real Dense, BM25, Hybrid, and Reranker services."""
        # 1. Dense
        vector_store = VectorStore()
        embedder = EmbeddingService()
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            vector_store.load(FAISS_INDEX_PATH, METADATA_PATH)
        dense_service = DenseRetrievalService(
            embedding_service=embedder, vector_store=vector_store
        )

        # 2. BM25
        bm25_store = BM25Store()
        bm25_service = BM25RetrievalService(bm25_store=bm25_store)
        bm25_service.initialize(chunks_path=_CHUNKS_PATH)

        # 3. Hybrid
        hybrid_service = HybridRetrievalService(
            dense_service=dense_service, bm25_service=bm25_service
        )

        # 4. Reranker
        reranker_service = RerankerService()

        # 5. Reranked Hybrid Service
        service = RerankedHybridService(
            hybrid_service=hybrid_service, reranker_service=reranker_service
        )
        return service

    def test_reranked_service_initialization(
        self, reranked_service: RerankedHybridService
    ) -> None:
        """Verify underlying stores and reranker model initialize cleanly."""
        reranked_service.initialize()
        assert reranked_service.reranker_service.is_initialized is True

    def test_end_to_end_reranked_retrieval(
        self, reranked_service: RerankedHybridService
    ) -> None:
        """Query execution returns structured RetrievalResponse with 1-indexed ranks."""
        response = reranked_service.retrieve(
            "What are the service hours for the library module?", top_k=5
        )

        assert isinstance(response, RetrievalResponse)
        assert response.query == "What are the service hours for the library module?"
        assert response.top_k == 5
        assert len(response.results) == 5

        ranks = [r.rank for r in response.results]
        assert ranks == [1, 2, 3, 4, 5]

        for res in response.results:
            assert res.chunk_id.startswith("DOC")
            assert res.document_id.startswith("DOC")
            assert len(res.text) > 0

    def test_reranked_scores_strictly_descending(
        self, reranked_service: RerankedHybridService
    ) -> None:
        """Cross-Encoder relevance scores must be strictly non-increasing / descending."""
        response = reranked_service.retrieve(
            "unresolved critical incidents escalation rule", top_k=5
        )

        scores = [r.similarity_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_code_query_reranking(
        self, reranked_service: RerankedHybridService
    ) -> None:
        """Technical code query (NX-FAC-100) must rank target document at position 1 after reranking."""
        response = reranked_service.retrieve("NX-FAC-100", top_k=5)

        assert len(response.results) > 0
        top_res = response.results[0]
        assert top_res.rank == 1
        assert "DOC001" in top_res.document_id
