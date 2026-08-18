"""Unit tests for Phase 9 Liveness (/health) and Readiness (/ready) probe semantics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.api.dependencies import reset_services
from app.main import create_app


class TestHealthAndReadinessProbes:
    """Test /health and /ready endpoint behavior under healthy and degraded states."""

    def setup_method(self):
        reset_services()
        self.client = TestClient(create_app(), raise_server_exceptions=False)

    def teardown_method(self):
        reset_services()

    def test_health_liveness_always_200(self):
        """GET /health is a liveness probe that returns HTTP 200."""
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")
        assert "environment" in data
        assert "components" in data

    def test_ready_probe_200_when_all_initialized(self):
        """GET /ready returns HTTP 200 when all backend services are initialized."""
        mock_rag = MagicMock()
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = True
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = True
        mock_rag.hybrid_reranker.reranker_service.is_initialized = True

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.get("/ready")
                assert resp.status_code == 200
                data = resp.json()
                assert data["ready"] is True
                assert data["status"] == "ready"
                assert data["components"]["retrieval"] is True
                assert data["components"]["reranker"] is True
                assert data["components"]["generation"] is True
                assert data["components"]["citation_verifier"] is True
                assert "timestamp" in data

    def test_ready_probe_503_when_vector_store_uninitialized(self):
        """GET /ready returns HTTP 503 Service Unavailable when vector_store is not initialized."""
        mock_rag = MagicMock()
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = False
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = True
        mock_rag.hybrid_reranker.reranker_service.is_initialized = True

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.get("/ready")
                assert resp.status_code == 503
                data = resp.json()
                assert data["ready"] is False
                assert data["status"] == "not_ready"
                assert data["components"]["retrieval"] is False
                assert data["components"]["reranker"] is True

    def test_ready_probe_503_when_bm25_uninitialized(self):
        """GET /ready returns HTTP 503 when BM25 store is uninitialized."""
        mock_rag = MagicMock()
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = True
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = False
        mock_rag.hybrid_reranker.reranker_service.is_initialized = True

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.get("/ready")
                assert resp.status_code == 503
                data = resp.json()
                assert data["ready"] is False
                assert data["components"]["retrieval"] is False

    def test_ready_probe_503_when_cross_encoder_uninitialized(self):
        """GET /ready returns HTTP 503 when Cross-Encoder is uninitialized."""
        mock_rag = MagicMock()
        mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = True
        mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = True
        mock_rag.hybrid_reranker.reranker_service.is_initialized = False

        with patch("app.api.routes.get_rag_service", return_value=mock_rag):
            with patch("app.api.dependencies.get_rag_service", return_value=mock_rag):
                resp = self.client.get("/ready")
                assert resp.status_code == 503
                data = resp.json()
                assert data["ready"] is False
                assert data["components"]["reranker"] is False
