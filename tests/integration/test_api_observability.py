"""Integration tests for Phase 9 API Observability Middleware and access logging."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.citations.models import CitationVerificationResult
from app.main import create_app
from app.rag.models import RAGResponse, RAGSource


class TestAPIObservabilityIntegration:
    """Test full observability pipeline: middleware, logging handler, and security headers."""

    def setup_method(self):
        reset_services()
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def teardown_method(self):
        reset_services()

    def test_structured_log_emitted_on_request(self):
        """HTTP request triggers a structured JSON access log event."""
        log_records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        access_logger = logging.getLogger("app.api.access")
        access_logger.setLevel(logging.INFO)
        handler = ListHandler()
        access_logger.addHandler(handler)

        try:
            resp = self.client.get("/health")
            assert resp.status_code == 200
            assert len(log_records) >= 1
            rec = log_records[-1]
            data = getattr(rec, "structured_data", {})
            assert data.get("event") == "http_request"
            assert data.get("method") == "GET"
            assert data.get("path") == "/health"
            assert data.get("status_code") == 200
            assert "duration_ms" in data
            assert "request_id" in data
        finally:
            access_logger.removeHandler(handler)

    def test_query_request_does_not_log_raw_query_text(self):
        """Query request logs query_length and query_hash, but NOT raw query text."""
        log_records = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        access_logger = logging.getLogger("app.api.access")
        access_logger.setLevel(logging.INFO)
        handler = ListHandler()
        access_logger.addHandler(handler)

        source = RAGSource(
            chunk_id="DOC001_CHUNK_01",
            document_id="DOC001",
            title="Operations Policy",
            department="Operations",
            rank=1,
            score=0.95,
            text="Sample policy evidence text.",
        )
        mock_response = RAGResponse(
            query="Confidential user secret query text",
            answer="Answer text",
            sources=[source],
            verification=None,
            groundedness_score=1.0,
        )
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_response

        try:
            with patch("app.api.routes.get_rag_service", return_value=mock_rag):
                with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                    resp = self.client.post(
                        "/query",
                        json={"query": "Confidential user secret query text", "top_k": 5},
                    )
                    assert resp.status_code == 200

            assert len(log_records) >= 1
            rec = log_records[-1]
            data = getattr(rec, "structured_data", {})
            assert "Confidential user secret" not in str(data)
            assert data.get("method") == "POST"
            assert data.get("path") == "/query"
        finally:
            access_logger.removeHandler(handler)
