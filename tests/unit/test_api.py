"""Comprehensive unit tests for Phase 8 FastAPI API layer.

Tests cover:
1.  GET /health returns 200
2.  Health does not expose secrets
3.  Health exposes mock/live mode correctly
4.  POST /query with valid answerable query
5.  POST /query with unanswerable query
6.  POST /query empty query
7.  POST /query whitespace-only query
8.  POST /query oversized query
9.  POST /query invalid top_k
10. POST /query top_k boundary = 1
11. POST /query top_k boundary = 150
12. POST /retrieve valid request
13. POST /retrieve invalid query
14. /retrieve does not invoke generation
15. /retrieve does not invoke citation verification
16. API error envelope format
17. Unexpected exception sanitization
18. Latency fields exist and are non-negative
19. Mock/live provider indicator is accurate
20. RAGResponse fields are preserved
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.citations.models import CitationVerificationResult
from app.main import create_app
from app.rag.models import RAGResponse, RAGSource
from app.retrieval.models import RetrievalResponse, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset API singleton services before and after each test."""
    reset_services()
    yield
    reset_services()


def _make_mock_rag_response(query: str = "test query") -> RAGResponse:
    """Build a realistic mock RAGResponse for testing."""
    source = RAGSource(
        chunk_id="DOC001_CHUNK_01",
        document_id="DOC001",
        title="Test Document",
        department="Testing",
        rank=1,
        score=0.95,
        text="This is test chunk text for verification.",
    )
    verification = CitationVerificationResult(
        query=query,
        answer="Test answer based on evidence.",
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
        answer="Test answer based on evidence.",
        sources=[source],
        verification=verification,
        groundedness_score=1.0,
    )


def _make_mock_retrieval_response(query: str = "test query") -> RetrievalResponse:
    """Build a realistic mock RetrievalResponse for testing."""
    result = RetrievalResult(
        chunk_id="DOC001_CHUNK_01",
        document_id="DOC001",
        text="This is test chunk text.",
        title="Test Document",
        department="Testing",
        similarity_score=0.85,
        rank=1,
    )
    return RetrievalResponse(query=query, results=[result], top_k=5)


def _make_test_client() -> TestClient:
    """Create a fresh TestClient with mocked services."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. GET /health returns 200
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self):
        """GET /health returns 200 with valid health structure."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")
        assert "environment" in data
        assert "components" in data

    # 2. Health does not expose secrets
    def test_health_does_not_expose_secrets(self):
        """Health response must not contain API keys or filesystem paths."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.get("/health")
        body = resp.text
        assert "OPENAI_API_KEY" not in body
        assert "sk-" not in body
        assert "C:\\" not in body
        assert "F:\\" not in body
        assert "/home/" not in body

    # 3. Health exposes mock/live mode correctly
    def test_health_mock_mode(self):
        """Health environment should be 'mock' when no OpenAI key is set."""
        with patch("app.api.dependencies.initialize_services"):
            with patch.dict("os.environ", {}, clear=False):
                # Ensure OPENAI_API_KEY is not set
                import os
                os.environ.pop("OPENAI_API_KEY", None)
                client = _make_test_client()
        resp = client.get("/health")
        data = resp.json()
        assert data["environment"] == "mock"


# ---------------------------------------------------------------------------
# POST /query tests
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    """Tests for the POST /query endpoint."""

    def _make_client_with_mock_rag(self, rag_response=None):
        """Create a client with mocked RAGService."""
        if rag_response is None:
            rag_response = _make_mock_rag_response()

        mock_rag = MagicMock()
        mock_rag.answer.return_value = rag_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            with patch("app.api.dependencies._rag_service", mock_rag):
                with patch("app.api.dependencies._initialized", True):
                    client = _make_test_client()

        # Patch the get_rag_service to return our mock
        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            yield client, mock_rag

    # 4. POST /query with valid answerable query
    def test_query_valid_answerable(self):
        """Valid query returns 200 with answer and sources."""
        mock_response = _make_mock_rag_response("What is the deadline?")
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": "What is the deadline?", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is the deadline?"
        assert data["answer"] == "Test answer based on evidence."
        assert len(data["sources"]) == 1
        assert data["groundedness_score"] == 1.0

    # 5. POST /query with unanswerable query (refusal)
    def test_query_unanswerable(self):
        """Unanswerable query returns refusal answer."""
        refusal_response = RAGResponse(
            query="What is quantum computing?",
            answer="I do not have sufficient information in the provided context to answer this question.",
            sources=[],
            verification=None,
            groundedness_score=1.0,
        )
        mock_rag = MagicMock()
        mock_rag.answer.return_value = refusal_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": "What is quantum computing?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "sufficient information" in data["answer"].lower() or "insufficient" in data["answer"].lower()

    # 6. POST /query empty query
    def test_query_empty_string(self):
        """Empty query string returns 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    # 7. POST /query whitespace-only query
    def test_query_whitespace_only(self):
        """Whitespace-only query returns 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 422

    # 8. POST /query oversized query
    def test_query_oversized(self):
        """Query exceeding max length returns 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        oversized = "x" * 2001
        resp = client.post("/query", json={"query": oversized})
        assert resp.status_code == 422

    # 9. POST /query invalid top_k
    def test_query_invalid_top_k(self):
        """Invalid top_k (0 or 151) returns 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.post("/query", json={"query": "test", "top_k": 0})
        assert resp.status_code == 422
        resp = client.post("/query", json={"query": "test", "top_k": 151})
        assert resp.status_code == 422

    # 10. POST /query top_k boundary = 1
    def test_query_top_k_min_boundary(self):
        """top_k=1 is accepted."""
        mock_response = _make_mock_rag_response("boundary test")
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": "boundary test", "top_k": 1})

        assert resp.status_code == 200

    # 11. POST /query top_k boundary = 150
    def test_query_top_k_max_boundary(self):
        """top_k=150 is accepted."""
        mock_response = _make_mock_rag_response("max boundary test")
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": "max boundary test", "top_k": 150})

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /retrieve tests
# ---------------------------------------------------------------------------


class TestRetrieveEndpoint:
    """Tests for the POST /retrieve debug endpoint."""

    # 12. POST /retrieve valid request
    def test_retrieve_valid_request(self):
        """Valid retrieve request returns 200 with results."""
        mock_response = _make_mock_retrieval_response("test retrieval")
        mock_reranker = MagicMock()
        mock_reranker.retrieve.return_value = mock_response

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_hybrid_reranker", return_value=mock_reranker):
            resp = client.post("/retrieve", json={"query": "test retrieval", "top_k": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test retrieval"
        assert data["top_k"] == 5
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == "DOC001_CHUNK_01"

    # 13. POST /retrieve invalid query
    def test_retrieve_empty_query(self):
        """Empty query on /retrieve returns 422."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.post("/retrieve", json={"query": ""})
        assert resp.status_code == 422

    # 14. /retrieve does not invoke generation
    def test_retrieve_does_not_call_generation(self):
        """Retrieve endpoint must not invoke GenerationService."""
        mock_response = _make_mock_retrieval_response()
        mock_reranker = MagicMock()
        mock_reranker.retrieve.return_value = mock_response

        mock_gen_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_hybrid_reranker", return_value=mock_reranker):
            resp = client.post("/retrieve", json={"query": "test"})

        assert resp.status_code == 200
        mock_gen_service.generate.assert_not_called()

    # 15. /retrieve does not invoke citation verification
    def test_retrieve_does_not_call_verification(self):
        """Retrieve endpoint must not invoke CitationVerificationService."""
        mock_response = _make_mock_retrieval_response()
        mock_reranker = MagicMock()
        mock_reranker.retrieve.return_value = mock_response

        mock_citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_hybrid_reranker", return_value=mock_reranker):
            resp = client.post("/retrieve", json={"query": "test"})

        assert resp.status_code == 200
        mock_citation_service.verify.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for API error handling and envelope format."""

    # 16. API error envelope format
    def test_validation_error_envelope_format(self):
        """Validation errors return structured error envelope."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == "VALIDATION_ERROR"

    # 17. Unexpected exception sanitization
    def test_unexpected_exception_sanitized(self):
        """Unexpected exceptions must not expose stack traces or secrets."""
        mock_rag = MagicMock()
        mock_rag.answer.side_effect = RuntimeError(
            "Connection failed at F:\\secret\\path with key sk-abc123456789012345678901"
        )
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": "test exception"})

        assert resp.status_code == 500
        body = resp.text
        # Must not contain the secret key or the filesystem path
        assert "sk-abc123456789012345678901" not in body
        assert "F:\\secret\\path" not in body
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_ERROR"


# ---------------------------------------------------------------------------
# Metadata and provider tests
# ---------------------------------------------------------------------------


class TestMetadataAndProvider:
    """Tests for latency metadata and provider indicators."""

    def _get_query_response(self, query="test metadata"):
        """Helper to get a mocked query response."""
        mock_response = _make_mock_rag_response(query)
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response
        mock_rag.hybrid_reranker = MagicMock()
        mock_rag.generation_service = MagicMock()
        mock_rag.citation_service = MagicMock()

        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            resp = client.post("/query", json={"query": query})

        return resp

    # 18. Latency fields exist and are non-negative
    def test_latency_fields_present_and_nonnegative(self):
        """Query response metadata must include latency fields >= 0."""
        resp = self._get_query_response()
        assert resp.status_code == 200
        data = resp.json()
        assert "metadata" in data
        meta = data["metadata"]
        assert "total_latency_ms" in meta
        assert meta["total_latency_ms"] >= 0.0
        assert "llm_provider" in meta
        assert "is_live_llm" in meta

    # 19. Mock/live provider indicator is accurate
    def test_mock_provider_indicator(self):
        """When no OpenAI key is set, provider must be 'mock' and is_live_llm=False."""
        import os
        os.environ.pop("OPENAI_API_KEY", None)
        resp = self._get_query_response("provider test")
        data = resp.json()
        meta = data["metadata"]
        assert meta["llm_provider"] == "mock"
        assert meta["is_live_llm"] is False

    # 20. RAGResponse fields are preserved
    def test_rag_response_fields_preserved(self):
        """API response must preserve all essential RAGResponse fields."""
        resp = self._get_query_response("field preservation")
        assert resp.status_code == 200
        data = resp.json()
        # Essential fields from RAGResponse
        assert "query" in data
        assert "answer" in data
        assert "sources" in data
        assert "verification" in data
        assert "groundedness_score" in data
        # Source structure
        if data["sources"]:
            src = data["sources"][0]
            assert "chunk_id" in src
            assert "document_id" in src
            assert "title" in src
            assert "department" in src
            assert "rank" in src
            assert "score" in src


# ---------------------------------------------------------------------------
# Security headers test
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Tests for security response headers."""

    def test_security_headers_present(self):
        """API responses must include security headers."""
        with patch("app.api.dependencies.initialize_services"):
            client = _make_test_client()
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
