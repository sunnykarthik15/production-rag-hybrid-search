"""Integration tests for Hybrid Retrieval Service using Reciprocal Rank Fusion (RRF).

Loads the actual FAISS dense vector store and BM25 store against the real 150-chunk dataset
and executes real query searches to verify end-to-end hybrid retrieval functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import FAISS_INDEX_PATH, METADATA_PATH
from app.embeddings.service import EmbeddingService
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
    reason="Chunks dataset not found. Skipping hybrid retrieval integration tests.",
)


class TestHybridRetrievalIntegration:
    """Integration test suite for hybrid search against full 150-chunk dataset."""

    @pytest.fixture(scope="class")
    def hybrid_service(self) -> HybridRetrievalService:
        """Initialize real Dense, BM25, and Hybrid services for integration tests."""
        # 1. Dense retrieval service
        vector_store = VectorStore()
        embedder = EmbeddingService()
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            vector_store.load(FAISS_INDEX_PATH, METADATA_PATH)
        dense_service = DenseRetrievalService(
            embedding_service=embedder, vector_store=vector_store
        )

        # 2. BM25 retrieval service
        bm25_store = BM25Store()
        bm25_service = BM25RetrievalService(bm25_store=bm25_store)
        bm25_service.initialize(chunks_path=_CHUNKS_PATH)

        # 3. Hybrid service
        service = HybridRetrievalService(
            dense_service=dense_service, bm25_service=bm25_service
        )
        return service

    def test_hybrid_service_initialization(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Verify hybrid service initializes cleanly and underlying stores are ready."""
        assert hybrid_service.dense_service.vector_store.is_initialized is True
        assert hybrid_service.bm25_service.bm25_store.is_initialized is True

    def test_retrieval_returns_valid_response_structure(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Query execution must return valid RetrievalResponse object."""
        response = hybrid_service.retrieve("routine maintenance SLA policy", top_k=5)

        assert isinstance(response, RetrievalResponse)
        assert response.query == "routine maintenance SLA policy"
        assert response.top_k == 5
        assert len(response.results) == 5

    def test_results_contain_valid_chunk_ids_and_ranks(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Retrieved items must contain valid chunk IDs and strict 1-indexed ranks."""
        response = hybrid_service.retrieve("building operations owner", top_k=5)

        ranks = [r.rank for r in response.results]
        assert ranks == [1, 2, 3, 4, 5]

        for res in response.results:
            assert res.chunk_id.startswith("DOC")
            assert res.document_id.startswith("DOC")
            assert len(res.text) > 0
            assert res.similarity_score > 0.0

    def test_results_scores_strictly_descending(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """RRF fused similarity scores must be strictly non-increasing / descending."""
        response = hybrid_service.retrieve("critical security incident response SLA", top_k=10)

        scores = [r.similarity_score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_keyword_query_retrieval(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Exact alphanumeric identifier query (NX-FAC-100) must retrieve corresponding chunk at rank 1."""
        response = hybrid_service.retrieve("NX-FAC-100", top_k=5)

        assert len(response.results) > 0
        top_result = response.results[0]
        assert top_result.rank == 1
        assert "DOC001" in top_result.document_id

    def test_semantic_query_retrieval(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Semantic query for library module service hours should rank library chunks highly."""
        response = hybrid_service.retrieve(
            "What are the service hours for the library module?", top_k=5
        )

        assert len(response.results) == 5
        top_docs = [r.document_id for r in response.results]
        # Library module documents in dataset are DOC002, DOC012, DOC022
        assert any(doc in top_docs for doc in ["DOC002", "DOC012", "DOC022"])

    def test_dense_and_bm25_fusion_benefits(
        self, hybrid_service: HybridRetrievalService
    ) -> None:
        """Query with both technical code and conceptual topic fuses results from both retrievers."""
        response = hybrid_service.retrieve(
            "NX-SEC-103 routine security operations queue SLA", top_k=10
        )

        assert len(response.results) == 10
        retrieved_ids = [r.chunk_id for r in response.results]
        # DOC004_CHUNK_01 or DOC004_CHUNK_05 contains NX-SEC-103
        assert any("DOC004" in cid for cid in retrieved_ids)
