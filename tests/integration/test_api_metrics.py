"""Integration tests for Phase 9 GET /metrics endpoint and live metric aggregation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.citations.models import CitationVerificationResult
from app.main import create_app
from app.observability.metrics import reset_metrics
from app.rag.models import RAGResponse, RAGSource
from app.retrieval.models import RetrievalResponse, RetrievalResult


class TestAPIMetricsIntegration:
    """Test /metrics endpoint accuracy across various request patterns and outcomes."""

    def setup_method(self):
        reset_services()
        reset_metrics()
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def teardown_method(self):
        reset_services()
        reset_metrics()

    def test_metrics_endpoint_initial_state(self):
        """GET /metrics returns well-structured empty metrics snapshot."""
        resp = self.client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests" in data
        assert "rag" in data
        assert "latency_ms" in data
        assert data["requests"]["total"] >= 0
        assert data["requests"]["error_rate"] == 0.0

    def test_metrics_increment_on_endpoint_calls(self):
        """Calling various endpoints correctly increments counters and endpoint breakdowns."""
        self.client.get("/health")
        self.client.get("/health")

        resp = self.client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["requests"]["by_endpoint"]["/health"] == 2

    def test_metrics_track_errors(self):
        """Failed requests increment failed_requests and errors_by_status."""
        self.client.get("/nonexistent-404")
        self.client.post("/retrieve", json={"query": ""})  # 422 Unprocessable Entity

        resp = self.client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["requests"]["failed"] >= 2
        assert data["requests"]["errors_by_status"].get("404", 0) >= 1
        assert data["requests"]["errors_by_status"].get("422", 0) >= 1

    def test_metrics_track_rag_subphases(self):
        """Executing /query records retrieval, LLM, verification, and groundedness metrics."""
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

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.post("/query", json={"query": "Test query", "top_k": 5})
                assert resp.status_code == 200

        metrics_resp = self.client.get("/metrics")
        data = metrics_resp.json()
        rag_metrics = data["rag"]
        assert rag_metrics["retrieval_executions"] >= 1
        assert rag_metrics["evidence_sufficient"] >= 1
        assert rag_metrics["llm_executions"] >= 1
        assert rag_metrics["citation_verifications"] >= 1
        assert rag_metrics["grounded_answers"] >= 1
        assert rag_metrics["groundedness_rate"] == 1.0
