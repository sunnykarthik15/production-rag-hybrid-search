"""Unit tests for BM25 lexical store and retrieval service."""

from __future__ import annotations

import pytest

from app.ingestion.models import Chunk
from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.bm25_store import BM25Store, BM25StoreError, tokenize
from app.retrieval.service import RetrievalError


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunk objects for BM25 unit tests."""
    return [
        Chunk(
            chunk_id="DOC001_CHUNK_01",
            document_id="DOC001",
            title="Facilities Operating Guide NX-FAC-100",
            department="Facilities",
            text="Service hours for facilities building requests are 08:00-17:00. SLA is 4 hours.",
            chunk_index=1,
            source="Test Engine",
        ),
        Chunk(
            chunk_id="DOC002_CHUNK_01",
            document_id="DOC002",
            title="Library Operations Manual NX-LIB-101",
            department="Library",
            text="Routine library service hours are 09:00-18:00 with SLA response window of 8 hours.",
            chunk_index=1,
            source="Test Engine",
        ),
        Chunk(
            chunk_id="DOC003_CHUNK_01",
            document_id="DOC003",
            title="Transport Logistics Rules NX-TRN-102",
            department="Transport",
            text="Shuttle schedule operates between 07:30-16:30. Escalation time is 30 minutes.",
            chunk_index=1,
            source="Test Engine",
        ),
    ]


class TestTokenize:
    """Test suite for tokenize helper function."""

    def test_lower_casing_and_punctuation(self) -> None:
        text = "Facilities SLA response time: 4-hours! (NX-FAC-100)"
        tokens = tokenize(text)
        assert "facilities" in tokens
        assert "sla" in tokens
        assert "nx-fac-100" in tokens
        assert "4-hours" in tokens

    def test_empty_string(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestBM25Store:
    """Test suite for BM25Store index building and searching."""

    def test_index_construction(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        assert not store.is_initialized
        assert store.chunk_count == 0

        store.build(sample_chunks)
        assert store.is_initialized
        assert store.chunk_count == 3

    def test_build_empty_raises(self) -> None:
        store = BM25Store()
        with pytest.raises(BM25StoreError, match="empty chunks list"):
            store.build([])

    def test_query_search_and_ranking(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)

        results = store.search("library operating hours", top_k=2)
        assert len(results) == 2
        top_meta, top_score, rank = results[0]
        assert rank == 1
        assert top_meta["chunk_id"] == "DOC002_CHUNK_01"
        assert top_meta["document_id"] == "DOC002"
        assert top_score > 0.0

    def test_top_k_behavior(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)

        results = store.search("service hours", top_k=1)
        assert len(results) == 1

        results_all = store.search("service hours", top_k=10)
        assert len(results_all) == 3

    def test_unknown_query_handling(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)

        # Query with non-matching terms should return results with zero scores
        results = store.search("quantum astrophysics photon collider", top_k=2)
        assert len(results) == 2
        for meta, score, rank in results:
            assert score == 0.0

    def test_uninitialized_search_raises(self) -> None:
        store = BM25Store()
        with pytest.raises(BM25StoreError, match="not initialized"):
            store.search("test")


class TestBM25RetrievalService:
    """Test suite for BM25RetrievalService API."""

    def test_service_search_returns_retrieval_response(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)
        service = BM25RetrievalService(bm25_store=store)

        response = service.search("facilities SLA", top_k=2)
        assert response.query == "facilities SLA"
        assert response.top_k == 2
        assert len(response.results) == 2
        assert response.results[0].chunk_id == "DOC001_CHUNK_01"
        assert response.results[0].rank == 1

    def test_empty_query_raises_retrieval_error(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)
        service = BM25RetrievalService(bm25_store=store)

        with pytest.raises(RetrievalError, match="Query string must not be empty"):
            service.search("")

        with pytest.raises(RetrievalError, match="Query string must not be empty"):
            service.search("   ")

    def test_invalid_top_k_raises_retrieval_error(self, sample_chunks: list[Chunk]) -> None:
        store = BM25Store()
        store.build(sample_chunks)
        service = BM25RetrievalService(bm25_store=store)

        with pytest.raises(RetrievalError, match="top_k must be an integer"):
            service.search("valid query", top_k=0)

        with pytest.raises(RetrievalError, match="top_k must be an integer"):
            service.search("valid query", top_k=999)
