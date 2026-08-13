"""Unit tests for HybridRetrievalService and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieval.hybrid_service import HybridRetrievalService
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import RetrievalError


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
        text=f"Content text for {chunk_id}",
        title="Test Document Title",
        department="Operations",
        similarity_score=score,
        rank=rank,
    )


class TestHybridRetrievalService:
    """Unit test suite for RRF fusion logic, candidate fetching, and parameter validation."""

    def test_basic_rrf_score_calculation(self) -> None:
        """1. Verify basic RRF score calculation formula 1/(rrf_k + rank)."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        # Dense: C1 (rank 1)
        dense_mock.retrieve.return_value = RetrievalResponse(
            query="test", results=[_make_res("C1", rank=1)], top_k=50
        )
        # BM25: empty
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="test", results=[], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("test", top_k=10, rrf_k=60, candidate_k=50)

        assert len(res.results) == 1
        expected_score = 1.0 / (60 + 1)
        assert abs(res.results[0].similarity_score - expected_score) < 1e-6

    def test_correct_1_indexed_ranking(self) -> None:
        """2. Verify output ranks are strictly 1-indexed (1, 2, 3...)."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", rank=1), _make_res("C2", rank=2)], top_k=50
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C3", rank=1)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=3, candidate_k=50)

        ranks = [r.rank for r in res.results]
        assert ranks == [1, 2, 3]

    def test_same_chunk_in_dense_and_bm25(self) -> None:
        """3. Chunk appearing in both Dense and BM25 receives score contributions from both."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        # C1 is rank 1 in Dense, rank 2 in BM25
        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", rank=1)], top_k=50
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", rank=2)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=5, rrf_k=60, candidate_k=50)

        assert len(res.results) == 1
        expected_score = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
        assert abs(res.results[0].similarity_score - expected_score) < 1e-6

    def test_chunk_appearing_only_in_dense(self) -> None:
        """4. Chunk appearing only in Dense list remains eligible and scored correctly."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("DENSE_ONLY", rank=1)], top_k=50
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=50)

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=5, rrf_k=60, candidate_k=50)

        assert len(res.results) == 1
        assert res.results[0].chunk_id == "DENSE_ONLY"
        assert abs(res.results[0].similarity_score - (1.0 / 61)) < 1e-6

    def test_chunk_appearing_only_in_bm25(self) -> None:
        """5. Chunk appearing only in BM25 list remains eligible and scored correctly."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=50)
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("BM25_ONLY", rank=1)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=5, rrf_k=60, candidate_k=50)

        assert len(res.results) == 1
        assert res.results[0].chunk_id == "BM25_ONLY"
        assert abs(res.results[0].similarity_score - (1.0 / 61)) < 1e-6

    def test_completely_disjoint_result_lists(self) -> None:
        """6. Disjoint Dense and BM25 lists are properly merged and sorted by single-source RRF scores."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C_DENSE_1", rank=1), _make_res("C_DENSE_2", rank=2)], top_k=50
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C_BM25_1", rank=1), _make_res("C_BM25_2", rank=2)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=10, candidate_k=50)

        assert len(res.results) == 4
        chunk_ids = [r.chunk_id for r in res.results]
        # Rank 1 items (C_BM25_1, C_DENSE_1) both have score 1/61, sorted by chunk_id ascending: C_BM25_1, C_DENSE_1
        assert chunk_ids[0] == "C_BM25_1"
        assert chunk_ids[1] == "C_DENSE_1"

    def test_identical_result_lists(self) -> None:
        """7. Identical Dense and BM25 lists receive doubled RRF score contributions."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        chunks = [_make_res("C1", rank=1), _make_res("C2", rank=2)]
        dense_mock.retrieve.return_value = RetrievalResponse(query="q", results=chunks, top_k=50)
        bm25_mock.retrieve.return_value = RetrievalResponse(query="q", results=chunks, top_k=50)

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=10, rrf_k=60, candidate_k=50)

        assert len(res.results) == 2
        assert res.results[0].chunk_id == "C1"
        assert abs(res.results[0].similarity_score - 2.0 / 61) < 1e-6
        assert res.results[1].chunk_id == "C2"
        assert abs(res.results[1].similarity_score - 2.0 / 62) < 1e-6

    def test_correct_top_k_truncation(self) -> None:
        """8. Output results list is strictly truncated to requested top_k."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_items = [_make_res(f"C{i}", rank=i) for i in range(1, 20)]
        dense_mock.retrieve.return_value = RetrievalResponse(query="q", results=dense_items, top_k=50)
        bm25_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=50)

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=5, candidate_k=50)

        assert len(res.results) == 5
        assert res.top_k == 5

    def test_correct_candidate_k_usage(self) -> None:
        """9. Sub-service retrieve calls receive the specified candidate_k parameter."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=30)
        bm25_mock.retrieve.return_value = RetrievalResponse(query="q", results=[], top_k=30)

        service = HybridRetrievalService(dense_mock, bm25_mock)
        service.retrieve("q", top_k=5, candidate_k=30)

        dense_mock.retrieve.assert_called_once_with("q", top_k=30)
        bm25_mock.retrieve.assert_called_once_with("q", top_k=30)

    def test_empty_query_rejection(self) -> None:
        """10. Empty or whitespace query raises RetrievalError."""
        service = HybridRetrievalService(MagicMock(), MagicMock())
        with pytest.raises(RetrievalError, match="empty"):
            service.retrieve("")
        with pytest.raises(RetrievalError, match="empty"):
            service.retrieve("   \n\t ")

    def test_invalid_top_k_rejection(self) -> None:
        """11. Invalid top_k (<1 or >150) raises RetrievalError."""
        service = HybridRetrievalService(MagicMock(), MagicMock())
        with pytest.raises(RetrievalError, match="top_k"):
            service.retrieve("query", top_k=0)
        with pytest.raises(RetrievalError, match="top_k"):
            service.retrieve("query", top_k=200)

    def test_invalid_candidate_k_rejection(self) -> None:
        """12. candidate_k less than top_k or >150 raises RetrievalError."""
        service = HybridRetrievalService(MagicMock(), MagicMock())
        with pytest.raises(RetrievalError, match="candidate_k"):
            service.retrieve("query", top_k=10, candidate_k=5)
        with pytest.raises(RetrievalError, match="candidate_k"):
            service.retrieve("query", top_k=10, candidate_k=200)

    def test_invalid_rrf_k_rejection(self) -> None:
        """13. rrf_k less than 1 raises RetrievalError."""
        service = HybridRetrievalService(MagicMock(), MagicMock())
        with pytest.raises(RetrievalError, match="rrf_k"):
            service.retrieve("query", rrf_k=0)

    def test_deterministic_tie_breaking(self) -> None:
        """14. Equal RRF scores are tie-broken deterministically by chunk_id ascending."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        # Both C_Z and C_A receive rank 1 in their respective single lists (score = 1/61 each)
        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C_Z", rank=1)], top_k=50
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C_A", rank=1)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=2, candidate_k=50)

        # C_A comes before C_Z due to alphabetical tie-breaking
        assert res.results[0].chunk_id == "C_A"
        assert res.results[1].chunk_id == "C_Z"

    def test_final_result_ordering(self) -> None:
        """15. Output results are strictly sorted by similarity_score (RRF score) descending."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q",
            results=[_make_res("C1", rank=1), _make_res("C2", rank=2), _make_res("C3", rank=3)],
            top_k=50,
        )
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q",
            results=[_make_res("C2", rank=1), _make_res("C3", rank=2)],
            top_k=50,
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=3, candidate_k=50)

        scores = [r.similarity_score for r in res.results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_independence_from_raw_scores(self) -> None:
        """16. RRF score calculation depends purely on rank positions, not raw similarity/BM25 scores."""
        dense_mock = MagicMock()
        bm25_mock = MagicMock()

        # Item with very low similarity score (0.01) at rank 1
        dense_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C1", score=0.01, rank=1)], top_k=50
        )
        # Item with very high BM25 score (999.0) at rank 1
        bm25_mock.retrieve.return_value = RetrievalResponse(
            query="q", results=[_make_res("C2", score=999.0, rank=1)], top_k=50
        )

        service = HybridRetrievalService(dense_mock, bm25_mock)
        res = service.retrieve("q", top_k=2, rrf_k=60, candidate_k=50)

        # Both items get identical RRF score contribution 1/(60+1) = 1/61
        assert res.results[0].similarity_score == res.results[1].similarity_score
        assert abs(res.results[0].similarity_score - 1.0 / 61) < 1e-6
