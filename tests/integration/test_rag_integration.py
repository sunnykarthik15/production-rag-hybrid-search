"""Integration tests for end-to-end RAG Service orchestration.

Connects real RerankedHybridService (FAISS + BM25 + CrossEncoder) with deterministic MockLLMProvider.
Does NOT require external API keys or live internet access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import FAISS_INDEX_PATH, METADATA_PATH
from app.embeddings.service import EmbeddingService
from app.generation.service import GenerationService, MockLLMProvider
from app.rag.models import RAGResponse
from app.rag.service import RAGService
from app.reranking.hybrid_reranker import RerankedHybridService
from app.reranking.service import RerankerService
from app.retrieval.bm25_service import BM25RetrievalService
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.service import DenseRetrievalService
from app.retrieval.vector_store import VectorStore

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"
_DATASET_PRESENT = _CHUNKS_PATH.exists()

pytestmark = pytest.mark.skipif(
    not _DATASET_PRESENT,
    reason="Chunks dataset not found. Skipping RAG integration tests.",
)


class TestRAGIntegration:
    """Integration test suite for RAG Service against real corpus."""

    @pytest.fixture(scope="class")
    def rag_service(self) -> RAGService:
        """Initialize real RerankedHybridService + MockLLMProvider."""
        vector_store = VectorStore()
        embedder = EmbeddingService()
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            vector_store.load(FAISS_INDEX_PATH, METADATA_PATH)
        dense_service = DenseRetrievalService(
            embedding_service=embedder, vector_store=vector_store
        )

        bm25_store = BM25Store()
        bm25_service = BM25RetrievalService(bm25_store=bm25_store)
        bm25_service.initialize(chunks_path=_CHUNKS_PATH)

        hybrid_service = HybridRetrievalService(
            dense_service=dense_service, bm25_service=bm25_service
        )
        reranker_service = RerankerService()
        hybrid_reranker = RerankedHybridService(
            hybrid_service=hybrid_service, reranker_service=reranker_service
        )

        generation_service = GenerationService(provider=MockLLMProvider())

        service = RAGService(
            hybrid_reranker=hybrid_reranker, generation_service=generation_service
        )
        return service

    def test_rag_service_initialization(self, rag_service: RAGService) -> None:
        """Verify RAGService initializes underlying retrieval and reranker services cleanly."""
        rag_service.initialize()
        assert rag_service.hybrid_reranker.reranker_service.is_initialized is True

    def test_end_to_end_rag_question_answering(
        self, rag_service: RAGService
    ) -> None:
        """Verify real RAG execution outputs RAGResponse with grounded answer and citations."""
        response = rag_service.answer("What are the service hours for the library module?", top_k=5)

        assert isinstance(response, RAGResponse)
        assert response.query == "What are the service hours for the library module?"
        assert len(response.answer) > 0
        assert len(response.sources) == 5

        ranks = [s.rank for s in response.sources]
        assert ranks == [1, 2, 3, 4, 5]

        for source in response.sources:
            assert source.chunk_id.startswith("DOC")
            assert source.document_id.startswith("DOC")
            assert len(source.title) > 0

    def test_exact_code_query_rag_sources(
        self, rag_service: RAGService
    ) -> None:
        """Technical code query (NX-FAC-100) retrieves target DOC001 chunk as top citation."""
        response = rag_service.answer("NX-FAC-100", top_k=5)

        assert len(response.sources) > 0
        top_source = response.sources[0]
        assert top_source.rank == 1
        assert "DOC001" in top_source.document_id

    def test_unanswerable_query_returns_insufficient_evidence(
        self, rag_service: RAGService
    ) -> None:
        """Unanswerable query (e.g. quantum propulsion in Nexora) triggers evidence sufficiency cutoff."""
        response = rag_service.answer("What is the quantum propulsion policy for interdimensional spaceflights?", top_k=5)

        assert isinstance(response, RAGResponse)
        assert response.answer == "I do not have sufficient information in the provided context to answer this question."
        assert len(response.sources) == 0
