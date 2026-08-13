"""Integration tests for BM25 lexical retrieval against real dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.bm25_store import BM25Store

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"
_QUESTIONS_PATH = _REPO_ROOT / "data" / "evaluation" / "questions.jsonl"

_DATASET_PRESENT = _CHUNKS_PATH.exists() and _QUESTIONS_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _DATASET_PRESENT,
    reason="Dataset files not found. Skipping BM25 integration tests.",
)


class TestBM25RetrievalIntegration:
    """Integration suite for BM25 indexing and searching against full 150-chunk dataset."""

    @pytest.fixture(scope="class")
    def bm25_service(self) -> BM25RetrievalService:
        """Initialize BM25 retrieval service against real chunks.jsonl dataset."""
        service = BM25RetrievalService()
        service.initialize(chunks_path=_CHUNKS_PATH)
        return service

    def test_real_dataset_chunk_count(self, bm25_service: BM25RetrievalService) -> None:
        """Verify all 150 chunks are indexed in BM25Store."""
        assert bm25_service.bm25_store.is_initialized
        assert bm25_service.bm25_store.chunk_count == 150

    def test_real_query_exact_identifier(self, bm25_service: BM25RetrievalService) -> None:
        """Querying exact module identifier 'NX-FAC-100' should rank DOC001 chunks top."""
        response = bm25_service.search("NX-FAC-100", top_k=5)
        assert len(response.results) == 5
        top = response.results[0]
        assert top.document_id == "DOC001"
        assert top.rank == 1

    def test_real_query_library_keyword(self, bm25_service: BM25RetrievalService) -> None:
        """Querying library terms should retrieve library document chunks."""
        response = bm25_service.search("library operating hours module", top_k=5)
        assert len(response.results) == 5
        retrieved_docs = [r.document_id for r in response.results]
        assert any(doc in retrieved_docs for doc in ["DOC002", "DOC012", "DOC022"])

    def test_evaluation_compatibility(self, bm25_service: BM25RetrievalService) -> None:
        """Verify service.search output works seamlessly for metric evaluation."""
        response = bm25_service.search("What is the response SLA for Facilities?", top_k=10)
        assert response.top_k == 10
        chunk_ids = [r.chunk_id for r in response.results]
        assert len(chunk_ids) == 10
        assert all(isinstance(cid, str) and cid.startswith("DOC") for cid in chunk_ids)
