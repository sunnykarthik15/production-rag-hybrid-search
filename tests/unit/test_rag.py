"""Unit tests for RAGService orchestration layer and Evidence Sufficiency logic."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.generation.models import GenerationResponse
from app.rag.models import RAGResponse
from app.rag.service import RAGError, RAGService
from app.retrieval.models import RetrievalResponse, RetrievalResult


def _make_res(
    chunk_id: str, title: str, text: str, rank: int = 1, score: float = 0.85
) -> RetrievalResult:
    """Helper factory for RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="DOC001",
        text=text,
        title=title,
        department="Operations",
        similarity_score=score,
        rank=rank,
    )


class TestRAGService:
    """Unit test suite for RAGService orchestration and Evidence Sufficiency."""

    def test_rag_end_to_end_flow(self) -> None:
        """RAGService retrieves chunks, checks evidence sufficiency, invokes generation, and maps sources."""
        hybrid_mock = MagicMock()
        gen_mock = MagicMock()

        chunks = [
            _make_res("DOC001_CHUNK_01", "Module A", "Operating hours are 9 AM to 5 PM.", rank=1, score=2.5),
            _make_res("DOC001_CHUNK_02", "Module A", "Emergency support is 24/7.", rank=2, score=1.8),
        ]
        hybrid_mock.retrieve.return_value = RetrievalResponse(query="q", results=chunks, top_k=2)
        gen_mock.generate.return_value = GenerationResponse(
            answer="Operating hours are 9 AM to 5 PM.", model_name="mock"
        )

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=gen_mock, min_relevance_score=0.0)
        response = service.answer("What are the operating hours?", top_k=2)

        assert isinstance(response, RAGResponse)
        assert response.query == "What are the operating hours?"
        assert response.answer == "Operating hours are 9 AM to 5 PM."
        assert len(response.sources) == 2
        gen_mock.generate.assert_called_once()

    def test_empty_retrieval_returns_insufficient_evidence(self) -> None:
        """Empty retrieval returns grounded insufficient information answer with no sources."""
        hybrid_mock = MagicMock()
        gen_mock = MagicMock()

        hybrid_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=5)

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=gen_mock)
        response = service.answer("Unknown topic query", top_k=5)

        assert isinstance(response, RAGResponse)
        assert "sufficient information" in response.answer.lower()
        assert len(response.sources) == 0
        gen_mock.generate.assert_not_called()

    def test_low_relevance_score_returns_insufficient_evidence_and_skips_llm(self) -> None:
        """Low top relevance score (< min_relevance_score) triggers insufficient evidence response and does NOT call LLM."""
        hybrid_mock = MagicMock()
        gen_mock = MagicMock()

        # Top candidate has negative relevance logit score (-1.5)
        low_score_chunks = [
            _make_res("DOC008_CHUNK_03", "Irrelevant Doc", "Random unanswerable text.", rank=1, score=-1.5),
            _make_res("DOC008_CHUNK_05", "Irrelevant Doc", "Other text.", rank=2, score=-3.2),
        ]
        hybrid_mock.retrieve.return_value = RetrievalResponse(query="q", results=low_score_chunks, top_k=2)

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=gen_mock, min_relevance_score=0.0)
        response = service.answer("Unanswerable question about alien life", top_k=2)

        assert isinstance(response, RAGResponse)
        assert response.answer == "I do not have sufficient information in the provided context to answer this question."
        assert len(response.sources) == 0
        # CRITICAL SAFETY: Verify LLM provider generate() was NOT called
        gen_mock.generate.assert_not_called()

    def test_threshold_boundary_behavior(self) -> None:
        """Test boundary behavior around configured min_relevance_score threshold."""
        hybrid_mock = MagicMock()
        gen_mock = MagicMock()
        gen_mock.generate.return_value = GenerationResponse(answer="Grounded answer.", model_name="mock")

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=gen_mock, min_relevance_score=0.0)

        # 1. Just below threshold (-0.001) -> Insufficient evidence, LLM skipped
        hybrid_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", "T", "Text", rank=1, score=-0.001)], top_k=1
        )
        res_below = service.answer("Query below threshold")
        assert "sufficient information" in res_below.answer.lower()
        assert len(res_below.sources) == 0
        gen_mock.generate.assert_not_called()

        # 2. At or above threshold (0.001) -> Evidence sufficient, LLM called
        hybrid_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", "T", "Text", rank=1, score=0.001)], top_k=1
        )
        res_above = service.answer("Query above threshold")
        assert res_above.answer == "Grounded answer."
        assert len(res_above.sources) == 1
        gen_mock.generate.assert_called_once()

    def test_empty_query_raises_rag_error(self) -> None:
        """Empty or whitespace query raises RAGError."""
        service = RAGService(MagicMock(), MagicMock())

        with pytest.raises(RAGError, match="empty"):
            service.answer("")

        with pytest.raises(RAGError, match="empty"):
            service.answer("   ")

    def test_invalid_top_k_raises_rag_error(self) -> None:
        """Invalid top_k (<1 or >150) raises RAGError."""
        service = RAGService(MagicMock(), MagicMock())

        with pytest.raises(RAGError, match="top_k"):
            service.answer("Valid query", top_k=0)

        with pytest.raises(RAGError, match="top_k"):
            service.answer("Valid query", top_k=200)

    def test_retrieval_failure_wrapped_in_rag_error(self) -> None:
        """Exceptions in retrieval layer are wrapped in RAGError."""
        hybrid_mock = MagicMock()
        hybrid_mock.retrieve.side_effect = RuntimeError("FAISS index failure")

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=MagicMock())

        with pytest.raises(RAGError, match="Retrieval step failed"):
            service.answer("Valid query")

    def test_generation_failure_wrapped_in_rag_error(self) -> None:
        """Exceptions in generation layer are wrapped in RAGError."""
        hybrid_mock = MagicMock()
        gen_mock = MagicMock()

        hybrid_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", "T", "Text", score=1.5)], top_k=1
        )
        gen_mock.generate.side_effect = RuntimeError("LLM API timeout")

        service = RAGService(hybrid_reranker=hybrid_mock, generation_service=gen_mock, min_relevance_score=0.0)

        with pytest.raises(RAGError, match="Generation step failed"):
            service.answer("Valid query")
