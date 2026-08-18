"""Unit tests for Phase 9 X-Request-ID propagation, validation, and generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.citations.models import CitationVerificationResult
from app.main import create_app
from app.observability.middleware import _get_or_generate_request_id
from app.rag.models import RAGResponse, RAGSource


class TestRequestIDMiddleware:
    """Test X-Request-ID validation, extraction, generation, and response header propagation."""

    def setup_method(self):
        reset_services()
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def teardown_method(self):
        reset_services()

    def test_auto_generate_request_id_when_missing(self):
        """When client provides no X-Request-ID header, a UUID4 string is auto-generated."""
        resp = self.client.get("/health")
        assert resp.status_code == 200
        req_id = resp.headers.get("X-Request-ID")
        assert req_id is not None
        assert len(req_id) >= 32

    def test_preserve_valid_custom_request_id(self):
        """When client provides a valid alphanumeric/hyphen/underscore request ID, it is preserved."""
        custom_id = "req-test-12345_ABC"
        resp = self.client.get("/health", headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_id

    def test_replace_malformed_request_id_with_special_characters(self):
        """When client provides an invalid request ID with injection chars, it is replaced with a safe UUID."""
        malicious_id = "req<script>alert(1)</script>"
        resp = self.client.get("/health", headers={"X-Request-ID": malicious_id})
        assert resp.status_code == 200
        assigned_id = resp.headers.get("X-Request-ID")
        assert assigned_id != malicious_id
        assert "<script>" not in assigned_id

    def test_replace_excessively_long_request_id(self):
        """When client provides a request ID longer than 64 characters, it is replaced."""
        long_id = "a" * 100
        resp = self.client.get("/health", headers={"X-Request-ID": long_id})
        assert resp.status_code == 200
        assigned_id = resp.headers.get("X-Request-ID")
        assert assigned_id != long_id
        assert len(assigned_id) < 64

    def test_request_id_present_on_404_error(self):
        """Error responses still contain X-Request-ID correlation headers."""
        resp = self.client.get("/nonexistent-endpoint")
        assert resp.status_code == 404
        assert "X-Request-ID" in resp.headers

    def test_request_id_attached_to_query_response_metadata(self):
        """POST /query attaches the correlated request_id to LatencyMetadata."""
        source = RAGSource(
            chunk_id="DOC001_CHUNK_01",
            document_id="DOC001",
            title="Operations Policy",
            department="Operations",
            rank=1,
            score=0.95,
            text="Sample policy evidence text.",
        )
        verification = CitationVerificationResult(
            query="Test query",
            answer="Test answer [DOC001_CHUNK_01].",
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
        mock_response = RAGResponse(
            query="Test query",
            answer="Test answer [DOC001_CHUNK_01].",
            sources=[source],
            verification=verification,
            groundedness_score=1.0,
        )

        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response

        custom_id = "trace-uuid-999"
        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.post(
                    "/query",
                    json={"query": "Test query", "top_k": 5},
                    headers={"X-Request-ID": custom_id},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["metadata"]["request_id"] == custom_id
                assert data["metadata"]["query_hash"] is not None
