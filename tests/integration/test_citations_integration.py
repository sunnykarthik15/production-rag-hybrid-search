"""Integration tests for end-to-end RAG Service with Citation Verification.

Connects real RerankedHybridService (FAISS + BM25 + CrossEncoder) with deterministic MockLLMProvider
and CitationVerificationService against the real Nexora corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.citations.models import AnswerType, CitationVerificationResult
from app.citations.service import CitationVerificationService
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
    reason="Chunks dataset not found. Skipping citation integration tests.",
)


class TestCitationsIntegration:
    """Integration test suite for RAG Service with Citation Verification."""

    @pytest.fixture(scope="class")
    def rag_service(self) -> RAGService:
        """Initialize real RerankedHybridService + MockLLMProvider + CitationVerificationService."""
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
        citation_service = CitationVerificationService()

        service = RAGService(
            hybrid_reranker=hybrid_reranker,
            generation_service=generation_service,
            citation_service=citation_service,
        )
        return service

    def test_end_to_end_rag_with_verification(self, rag_service: RAGService) -> None:
        """Verify real RAG execution outputs RAGResponse with verified citations and groundedness report."""
        rag_service.initialize()
        response = rag_service.answer("What are the service hours for the library module?", top_k=5)

        assert isinstance(response, RAGResponse)
        assert response.query == "What are the service hours for the library module?"
        assert len(response.answer) > 0
        assert len(response.sources) == 5

        # Check source text preservation
        for s in response.sources:
            assert s.text is not None
            assert len(s.text) > 0

        # Check verification report
        assert response.verification is not None
        assert isinstance(response.verification, CitationVerificationResult)
        assert response.groundedness_score is not None
        assert response.verification.total_claims >= 1
        assert response.verification.grounded_claims_count >= 1

    def test_exact_code_query_with_verification(self, rag_service: RAGService) -> None:
        """Technical code query (NX-FAC-100) retrieves DOC001 and produces verified answer."""
        response = rag_service.answer("NX-FAC-100", top_k=5)

        assert response.verification is not None
        assert len(response.sources) > 0
        assert response.sources[0].rank == 1
        assert "DOC001" in response.sources[0].document_id

    def test_unanswerable_query_refusal_verification(
        self, rag_service: RAGService
     ) -> None:
        """Unanswerable query triggers evidence sufficiency cutoff and produces verified refusal."""
        response = rag_service.answer(
            "What is the quantum propulsion policy for interdimensional spaceflights?", top_k=5
        )

        assert isinstance(response, RAGResponse)
        assert "sufficient information" in response.answer.lower()
        assert response.verification is not None
        assert response.verification.answer_type == AnswerType.INSUFFICIENT_EVIDENCE
        assert response.verification.refusal_correct is True
        assert response.verification.claim_groundedness_rate == 1.0
        assert response.verification.total_claims == 0
        assert response.verification.spurious_citations_count == 0
        assert response.verification.citation_precision is None
        assert response.verification.citation_f1 is None

    def test_end_to_end_rag_with_generated_citations(
        self, rag_service: RAGService
     ) -> None:
        """Verify end-to-end RAG with citation generation enabled produces 100% verified citations."""
        citation_gen_service = GenerationService(
            provider=MockLLMProvider(include_citations=True)
        )
        cited_rag = RAGService(
            hybrid_reranker=rag_service.hybrid_reranker,
            generation_service=citation_gen_service,
            citation_service=rag_service.citation_service,
        )
        cited_rag.initialize()

        response = cited_rag.answer("What are the service hours for the library module?", top_k=5)

        assert isinstance(response, RAGResponse)
        assert "[Source 1]" in response.answer
        assert response.verification is not None
        assert response.verification.total_citations >= 1
        assert response.verification.correct_citations_count >= 1
        assert response.verification.citation_precision == 1.0
        assert response.verification.citation_recall == 1.0
        assert response.verification.citation_f1 == 1.0
        assert response.verification.is_fully_cited is True
        assert response.verification.missing_citations_count == 0


