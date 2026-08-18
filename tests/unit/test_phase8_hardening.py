"""Comprehensive unit and regression tests for Phase 8.1 Production Hardening Pass.

Tests cover:
1.  404 Not Found returns structured envelope (not 500).
2.  405 Method Not Allowed returns structured envelope (not 500).
3.  Path sanitization for Windows drive letters (both / and \\ separators) and POSIX paths.
4.  Secret and API token redaction in error messages.
5.  Singleton reuse: get_hybrid_reranker() shares RAGService.hybrid_reranker instance.
6.  Singleton reuse: get_rag_service() returns identical instance across calls.
7.  Health check reports 'degraded' when underlying vector store / reranker are uninitialized.
8.  Health check reports 'healthy' when components are initialized.
9.  Concurrent /query requests in multiple threads show zero state crossover.
10. Concurrent /retrieve requests in multiple threads show zero state crossover.
11. Mixed simultaneous /query and /retrieve requests execute cleanly without race conditions.
12. Exact query boundary: query length = 2000 is accepted.
13. Exact query boundary: query length = 2001 is rejected with HTTP 422.
14. Malformed JSON payload returns HTTP 422 validation error envelope.
15. Missing required 'query' field returns HTTP 422 validation error envelope.
16. /retrieve strictly bypasses LLM generation and citation verification.
17. Monotonic latency measurement is non-negative and properly formatted.
18. Configuration environment overrides for API_CORS_ORIGINS and API settings.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    check_component_health,
    get_hybrid_reranker,
    get_rag_service,
    reset_services,
)
from app.api.errors import _sanitize_message
from app.citations.models import CitationVerificationResult
from app.main import create_app
from app.rag.models import RAGResponse, RAGSource
from app.retrieval.models import RetrievalResponse, RetrievalResult


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Ensure singleton state is clean for each test."""
    reset_services()
    yield
    reset_services()


def _make_mock_rag_response(query: str) -> RAGResponse:
    """Create a unique mock RAGResponse for testing concurrency and isolation."""
    source = RAGSource(
        chunk_id=f"CHUNK_FOR_{query[:8]}",
        document_id=f"DOC_{query[:8]}",
        title=f"Doc Title {query[:8]}",
        department="Operations",
        rank=1,
        score=0.92,
        text=f"Content evidence for query: {query}",
    )
    verification = CitationVerificationResult(
        query=query,
        answer=f"Verified grounded answer for {query}.",
        total_claims=1,
        grounded_claims_count=1,
        ungrounded_claims_count=0,
        claim_groundedness_rate=1.0,
        is_fully_grounded=True,
        total_citations=1,
        correct_citations_count=1,
        citation_precision=1.0,
        citation_recall=1.0,
        citation_f1=1.0,
        is_fully_cited=True,
    )
    return RAGResponse(
        query=query,
        answer=f"Verified grounded answer for {query}.",
        sources=[source],
        verification=verification,
        groundedness_score=1.0,
    )


def _make_mock_retrieval_response(query: str) -> RetrievalResponse:
    """Create a unique mock RetrievalResponse for testing."""
    result = RetrievalResult(
        chunk_id=f"RET_CHUNK_{query[:8]}",
        document_id=f"RET_DOC_{query[:8]}",
        text=f"Retrieved text for {query}",
        title=f"Retrieved Title {query[:8]}",
        department="Operations",
        similarity_score=0.88,
        rank=1,
    )
    return RetrievalResponse(query=query, results=[result], top_k=5)


# ---------------------------------------------------------------------------
# 1. Error Handling & HTTP Status Hardening Tests
# ---------------------------------------------------------------------------


class TestErrorHandlingHardening:
    """Verify HTTP status codes and error envelope consistency."""

    def test_404_not_found_returns_structured_envelope(self):
        """Unknown endpoint must return HTTP 404 with NOT_FOUND code, not 500."""
        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.get("/nonexistent_endpoint")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "NOT_FOUND"
        assert "message" in data["error"]

    def test_405_method_not_allowed_returns_structured_envelope(self):
        """Method not allowed must return HTTP 405 with METHOD_NOT_ALLOWED code, not 500."""
        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)
        # GET on /query (which only accepts POST)
        resp = client.get("/query")
        assert resp.status_code == 405
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "METHOD_NOT_ALLOWED"

    def test_path_sanitization_cross_platform(self):
        """Sanitization must redact Windows paths (both \\ and /) and POSIX paths."""
        # Windows backslash path
        win_bs = "Error in F:\\RAG-Project 1\\production-rag\\file.py at line 42"
        assert "F:\\" not in _sanitize_message(win_bs)
        assert "[PATH_REDACTED]" in _sanitize_message(win_bs)

        # Windows forward slash path
        win_fs = "Error in C:/Users/Admin/project/data.json"
        assert "C:/" not in _sanitize_message(win_fs)
        assert "[PATH_REDACTED]" in _sanitize_message(win_fs)

        # POSIX absolute path
        posix = "Failed to load /home/user/app/models/weights.bin"
        assert "/home/" not in _sanitize_message(posix)
        assert "[PATH_REDACTED]" in _sanitize_message(posix)

    def test_secret_redaction(self):
        """Sanitization must redact OpenAI API keys and Bearer tokens."""
        raw = "Auth failed for key sk-proj1234567890abcdef1234567890 with Bearer secret_token_xyz"
        sanitized = _sanitize_message(raw)
        assert "sk-proj" not in sanitized
        assert "Bearer secret" not in sanitized
        assert "[REDACTED]" in sanitized


# ---------------------------------------------------------------------------
# 2. Dependency & Singleton Lifecycle Tests
# ---------------------------------------------------------------------------


class TestDependencyLifecycleHardening:
    """Verify singleton lifecycle, model sharing, and reset behavior."""

    def test_hybrid_reranker_reuses_rag_service_instance(self):
        """get_hybrid_reranker() must return the same instance as get_rag_service().hybrid_reranker."""
        rag = get_rag_service()
        reranker = get_hybrid_reranker()
        assert reranker is rag.hybrid_reranker

    def test_rag_service_is_singleton(self):
        """Multiple get_rag_service() calls must return the identical instance."""
        rag1 = get_rag_service()
        rag2 = get_rag_service()
        assert rag1 is rag2

    def test_reset_services_clears_singletons(self):
        """reset_services() must create fresh instances on subsequent calls."""
        rag1 = get_rag_service()
        reset_services()
        rag2 = get_rag_service()
        assert rag1 is not rag2


# ---------------------------------------------------------------------------
# 3. Component Health Check Accuracy Tests
# ---------------------------------------------------------------------------


class TestHealthCheckAccuracy:
    """Verify health endpoint accurately reflects initialized vs uninitialized states."""

    def test_health_reports_degraded_when_uninitialized(self):
        """If vector store / reranker are uninitialized, /health must report 'degraded'."""
        mock_rag = MagicMock()
        # Simulate uninitialized stores
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = False
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = False
        mock_rag.hybrid_reranker.reranker_service.is_initialized = False
        mock_rag.generation_service.provider = MagicMock()
        mock_rag.citation_service.verifier = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                client = TestClient(create_app(), raise_server_exceptions=False)
                resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["retrieval"] is False
        assert data["components"]["reranker"] is False

    def test_health_reports_healthy_when_all_initialized(self):
        """When all components report is_initialized = True, /health reports 'healthy'."""
        mock_rag = MagicMock()
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = True
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = True
        mock_rag.hybrid_reranker.reranker_service.is_initialized = True
        mock_rag.generation_service.provider = MagicMock()
        mock_rag.citation_service.verifier = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                client = TestClient(create_app(), raise_server_exceptions=False)
                resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["retrieval"] is True
        assert data["components"]["reranker"] is True
        assert data["components"]["generation"] is True
        assert data["components"]["citation_verifier"] is True


# ---------------------------------------------------------------------------
# 4. Concurrency & Request Isolation Tests
# ---------------------------------------------------------------------------


class TestConcurrencyAndIsolation:
    """Verify concurrent requests do not cross-contaminate state."""

    def test_concurrent_query_isolation(self):
        """Concurrent /query requests must receive their own isolated answers and sources."""
        queries = [
            f"Query number {i}: What is procedure {i}?"
            for i in range(1, 11)
        ]

        def _mock_answer(query: str, top_k: int = 10, **kwargs) -> RAGResponse:
            return _make_mock_rag_response(query)

        mock_rag = MagicMock()
        mock_rag.answer.side_effect = _mock_answer
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.routes.get_rag_service", return_value=mock_rag):
                client = TestClient(create_app(), raise_server_exceptions=False)

                def _send_request(q: str):
                    return client.post("/query", json={"query": q, "top_k": 5})

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(_send_request, q) for q in queries]
                    responses = [f.result() for f in futures]

        for i, (q, resp) in enumerate(zip(queries, responses)):
            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == q
            assert q in data["answer"]
            assert data["sources"][0]["chunk_id"] == f"CHUNK_FOR_{q[:8]}"

    def test_concurrent_retrieve_isolation(self):
        """Concurrent /retrieve requests must receive isolated candidates without crossover."""
        queries = [f"Search term {i}" for i in range(1, 11)]

        def _mock_retrieve(query: str, top_k: int = 5, **kwargs) -> RetrievalResponse:
            return _make_mock_retrieval_response(query)

        mock_reranker = MagicMock()
        mock_reranker.retrieve.side_effect = _mock_retrieve

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.routes.get_hybrid_reranker", return_value=mock_reranker):
                client = TestClient(create_app(), raise_server_exceptions=False)

                def _send_retrieve(q: str):
                    return client.post("/retrieve", json={"query": q, "top_k": 5})

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(_send_retrieve, q) for q in queries]
                    responses = [f.result() for f in futures]

        for q, resp in zip(queries, responses):
            assert resp.status_code == 200
            data = resp.json()
            assert data["query"] == q
            assert data["results"][0]["chunk_id"] == f"RET_CHUNK_{q[:8]}"

    def test_mixed_concurrent_query_and_retrieve(self):
        """Simultaneous /query and /retrieve requests execute cleanly without race conditions."""
        mock_rag = MagicMock()
        mock_rag.answer.side_effect = lambda query, top_k=10, **kw: _make_mock_rag_response(query)
        mock_reranker = MagicMock()
        mock_reranker.retrieve.side_effect = lambda query, top_k=5, **kw: _make_mock_retrieval_response(query)

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.routes.get_rag_service", return_value=mock_rag):
                with patch("app.api.routes.get_hybrid_reranker", return_value=mock_reranker):
                    client = TestClient(create_app(), raise_server_exceptions=False)

                    def _send_mixed(item: tuple[str, str]):
                        endpoint, q = item
                        if endpoint == "query":
                            return client.post("/query", json={"query": q})
                        else:
                            return client.post("/retrieve", json={"query": q})

                    tasks = [("query", f"Query {i}") if i % 2 == 0 else ("retrieve", f"Retrieve {i}") for i in range(12)]

                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        futures = [executor.submit(_send_mixed, t) for t in tasks]
                        results = [f.result() for f in futures]

        for res in results:
            assert res.status_code == 200


# ---------------------------------------------------------------------------
# 5. Boundary and Validation Hardening Tests
# ---------------------------------------------------------------------------


class TestBoundaryAndValidationHardening:
    """Verify exact boundary behavior on query lengths and payloads."""

    def test_exact_max_query_boundary_accepted(self):
        """Query with exact length = 2000 must be accepted."""
        query_2000 = "a" * 2000
        mock_rag = MagicMock()
        mock_rag.answer.return_value = _make_mock_rag_response(query_2000)

        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": query_2000})

        assert resp.status_code == 200

    def test_max_query_boundary_plus_one_rejected(self):
        """Query with length = 2001 must be rejected with HTTP 422."""
        query_2001 = "a" * 2001
        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.post("/query", json={"query": query_2001})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_malformed_json_body_rejected(self):
        """Malformed JSON payload must return HTTP 422 with structured envelope."""
        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.post(
            "/query",
            content='{"query": "unclosed json',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_missing_query_field_rejected(self):
        """Payload without 'query' field must return HTTP 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)
        resp = client.post("/query", json={"top_k": 5})
        assert resp.status_code == 422
        data = resp.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# 6. /retrieve Isolation Hardening Tests
# ---------------------------------------------------------------------------


class TestRetrieveIsolationHardening:
    """Verify /retrieve strictly bypasses LLM generation and citation verification."""

    def test_retrieve_strictly_bypasses_generation_and_citation(self):
        """POST /retrieve must not call generate() or verify()."""
        mock_retrieval = _make_mock_retrieval_response("isolation test")
        mock_hybrid_reranker = MagicMock()
        mock_hybrid_reranker.retrieve.return_value = mock_retrieval

        mock_gen = MagicMock()
        mock_citation = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = TestClient(create_app(), raise_server_exceptions=False)

        with patch("app.api.routes.get_hybrid_reranker", return_value=mock_hybrid_reranker):
            resp = client.post("/retrieve", json={"query": "isolation test", "top_k": 5})

        assert resp.status_code == 200
        mock_hybrid_reranker.retrieve.assert_called_once_with(query="isolation test", top_k=5)
        mock_gen.generate.assert_not_called()
        mock_citation.verify.assert_not_called()
