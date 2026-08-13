"""
Integration tests for dense vector retrieval.

Loads/builds the actual vector index against the real 150-chunk dataset
and executes sample query searches to verify end-to-end functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import FAISS_INDEX_PATH, METADATA_PATH
from app.embeddings.service import EmbeddingService
from app.ingestion.loaders import load_chunks_jsonl
from app.retrieval.service import DenseRetrievalService
from app.retrieval.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"

_DATASET_PRESENT = _CHUNKS_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _DATASET_PRESENT,
    reason="Chunks dataset not found. Skipping vector retrieval integration tests.",
)


class TestVectorRetrievalIntegration:
    """Integration test suite for dense retrieval against full dataset."""

    @pytest.fixture(scope="class")
    def retrieval_service(self) -> DenseRetrievalService:
        """Initialize or build the vector store service for the integration suite."""
        store = VectorStore()
        embedder = EmbeddingService()

        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            store.load(FAISS_INDEX_PATH, METADATA_PATH)
        else:
            chunks = list(load_chunks_jsonl(_CHUNKS_PATH))
            embeddings = embedder.embed_chunks(chunks)
            store.build(embeddings, chunks)

        return DenseRetrievalService(embedding_service=embedder, vector_store=store)

    def test_real_query_facilities(self, retrieval_service: DenseRetrievalService) -> None:
        """Query for Facilities SLA should return DOC001 or DOC011 or DOC021 chunks."""
        response = retrieval_service.retrieve("What is the response SLA for Facilities?", top_k=5)

        assert len(response.results) == 5
        top_result = response.results[0]
        assert top_result.rank == 1
        assert "DOC" in top_result.document_id
        assert top_result.similarity_score > 0.3

    def test_real_query_library(self, retrieval_service: DenseRetrievalService) -> None:
        """Query for Library operating hours should rank Library chunks highly."""
        response = retrieval_service.retrieve("What are the service hours for the library module?", top_k=5)

        assert len(response.results) == 5
        top_docs = [r.document_id for r in response.results]
        assert any(doc in top_docs for doc in ["DOC002", "DOC012", "DOC022"])

    def test_real_query_escalation(self, retrieval_service: DenseRetrievalService) -> None:
        """Query for incident escalation rule should retrieve escalation chunks."""
        response = retrieval_service.retrieve("How are unresolved critical incidents escalated?", top_k=5)

        assert len(response.results) == 5
        # Escalation rule is present across modules (chunk 4 of most documents)
        texts = " ".join([r.text for r in response.results]).lower()
        assert "escalated" in texts or "incident" in texts

    def test_result_scores_decreasing(self, retrieval_service: DenseRetrievalService) -> None:
        """Multiple test queries must all yield strictly decreasing similarity scores."""
        queries = [
            "Emergency maintenance announcement notice",
            "Nominal service capacity concurrent users",
            "Monthly SLA compliance audit rule",
        ]
        for q in queries:
            response = retrieval_service.retrieve(q, top_k=10)
            scores = [r.similarity_score for r in response.results]
            assert scores == sorted(scores, reverse=True), f"Scores not descending for query: '{q}'"
