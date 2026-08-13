"""Unit tests for RerankerService and RerankedHybridService."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.reranking.hybrid_reranker import RerankedHybridService
from app.reranking.service import RerankerError, RerankerService
from app.retrieval.models import RetrievalResponse, RetrievalResult


def _make_res(
    chunk_id: str,
    doc_id: str = "DOC001",
    rank: int = 1,
    score: float = 0.5,
) -> RetrievalResult:
    """Helper factory for building dummy RetrievalResult objects."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"Sample text body for {chunk_id}",
        title="Sample Module Title",
        department="Operations",
        similarity_score=score,
        rank=rank,
    )


class TestRerankerService:
    """Unit tests for RerankerService."""

    def test_reranker_resorts_by_cross_encoder_scores(self) -> None:
        """RerankerService re-sorts candidates according to CrossEncoder output scores."""
        service = RerankerService()
        mock_model = MagicMock()
        # Mock predict returns higher score for item 2 (C2) than item 1 (C1)
        mock_model.predict.return_value = [1.2, 5.8, 0.4]
        service._model = mock_model

        candidates = [
            _make_res("C1", rank=1),
            _make_res("C2", rank=2),
            _make_res("C3", rank=3),
        ]

        response = service.rerank("test query", candidates, top_k=3)

        assert len(response.results) == 3
        # C2 should be ranked 1 because score is 5.8
        assert response.results[0].chunk_id == "C2"
        assert response.results[0].rank == 1
        assert response.results[0].similarity_score == 5.8

        # C1 should be ranked 2 because score is 1.2
        assert response.results[1].chunk_id == "C1"
        assert response.results[1].rank == 2
        assert response.results[1].similarity_score == 1.2

        # C3 should be ranked 3 because score is 0.4
        assert response.results[2].chunk_id == "C3"
        assert response.results[2].rank == 3
        assert response.results[2].similarity_score == 0.4

    def test_empty_candidates_returns_empty_response(self) -> None:
        """Reranking an empty candidate list returns an empty RetrievalResponse."""
        service = RerankerService()
        response = service.rerank("query text", candidates=[], top_k=5)

        assert isinstance(response, RetrievalResponse)
        assert len(response.results) == 0
        assert response.top_k == 5

    def test_empty_query_raises_reranker_error(self) -> None:
        """Empty query text raises RerankerError."""
        service = RerankerService()
        with pytest.raises(RerankerError, match="empty"):
            service.rerank("", [_make_res("C1")])
        with pytest.raises(RerankerError, match="empty"):
            service.rerank("   \t ", [_make_res("C1")])

    def test_invalid_top_k_raises_reranker_error(self) -> None:
        """Invalid top_k (<1 or >150) raises RerankerError."""
        service = RerankerService()
        with pytest.raises(RerankerError, match="top_k"):
            service.rerank("query", [_make_res("C1")], top_k=0)
        with pytest.raises(RerankerError, match="top_k"):
            service.rerank("query", [_make_res("C1")], top_k=200)

    def test_deterministic_tie_breaking(self) -> None:
        """Equal CrossEncoder scores are tie-broken deterministically by chunk_id ascending."""
        service = RerankerService()
        mock_model = MagicMock()
        # Both items receive identical score 2.5
        mock_model.predict.return_value = [2.5, 2.5]
        service._model = mock_model

        candidates = [_make_res("CHUNK_Z"), _make_res("CHUNK_A")]
        response = service.rerank("query", candidates, top_k=2)

        # CHUNK_A comes first alphabetically
        assert response.results[0].chunk_id == "CHUNK_A"
        assert response.results[1].chunk_id == "CHUNK_Z"

    def test_item_metadata_preserved(self) -> None:
        """All item metadata fields (doc_id, text, title, dept) are preserved after reranking."""
        service = RerankerService()
        mock_model = MagicMock()
        mock_model.predict.return_value = [3.0]
        service._model = mock_model

        original = RetrievalResult(
            chunk_id="DOC005_CHUNK_02",
            document_id="DOC005",
            text="Detailed operational policy guidelines.",
            title="Security Module 05",
            department="Security Operations",
            similarity_score=0.45,
            rank=4,
        )

        response = service.rerank("query", [original], top_k=1)
        res = response.results[0]

        assert res.chunk_id == "DOC005_CHUNK_02"
        assert res.document_id == "DOC005"
        assert res.text == "Detailed operational policy guidelines."
        assert res.title == "Security Module 05"
        assert res.department == "Security Operations"
        assert res.rank == 1
        assert res.similarity_score == 3.0


class TestRerankedHybridService:
    """Unit tests for RerankedHybridService integration container."""

    def test_reranked_hybrid_pipeline_flow(self) -> None:
        """RerankedHybridService calls HybridRetrievalService then RerankerService."""
        hybrid_mock = MagicMock()
        reranker_mock = MagicMock()

        candidates = [_make_res("C1"), _make_res("C2")]
        hybrid_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=candidates, top_k=30
        )
        reranker_mock.rerank.return_value = RetrievalResponse(
            query="q", results=[_make_res("C2", rank=1)], top_k=5
        )

        service = RerankedHybridService(
            hybrid_service=hybrid_mock, reranker_service=reranker_mock
        )

        response = service.retrieve("q", top_k=5, rrf_k=60, candidate_k=30)

        hybrid_mock.retrieve.assert_called_once_with(
            query="q", top_k=30, rrf_k=60, candidate_k=50
        )
        reranker_mock.rerank.assert_called_once_with(
            query="q", candidates=candidates, top_k=5
        )
        assert len(response.results) == 1
        assert response.results[0].chunk_id == "C2"

    def test_invalid_parameters_raise_reranker_error(self) -> None:
        """Invalid query, top_k, candidate_k, or rrf_k raise RerankerError."""
        service = RerankedHybridService(MagicMock(), MagicMock())

        with pytest.raises(RerankerError, match="empty"):
            service.retrieve("")

        with pytest.raises(RerankerError, match="top_k"):
            service.retrieve("query", top_k=0)

        with pytest.raises(RerankerError, match="candidate_k"):
            service.retrieve("query", top_k=10, candidate_k=5)

        with pytest.raises(RerankerError, match="rrf_k"):
            service.retrieve("query", rrf_k=0)
