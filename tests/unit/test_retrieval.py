"""Unit tests for app.retrieval module (VectorStore and DenseRetrievalService)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from app.config import EMBEDDING_DIM
from app.embeddings.service import EmbeddingService
from app.ingestion.models import Chunk
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import DenseRetrievalService, RetrievalError
from app.retrieval.vector_store import (
    IndexNotFoundError,
    VectorStore,
    VectorStoreError,
)


class TestVectorStore:
    """Unit tests for VectorStore."""

    def test_vector_index_creation(self, one_fifty_chunks: List[Chunk]) -> None:
        """Building index with normalized embeddings creates a valid store."""
        store = VectorStore(dimension=EMBEDDING_DIM)

        # Create dummy normalized random embeddings
        raw = np.random.randn(len(one_fifty_chunks), EMBEDDING_DIM).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norm_embeddings = raw / norms

        store.build(norm_embeddings, one_fifty_chunks)

        assert store.is_initialized is True
        assert store.size == 150

    def test_vector_index_save_and_load(
        self,
        tmp_path: Path,
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """Saving and reloading vector index must preserve entries and size."""
        store = VectorStore(dimension=EMBEDDING_DIM)

        raw = np.random.randn(len(one_fifty_chunks), EMBEDDING_DIM).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norm_embeddings = raw / norms

        store.build(norm_embeddings, one_fifty_chunks)

        idx_file = tmp_path / "test_index.faiss"
        meta_file = tmp_path / "test_metadata.json"

        store.save(idx_file, meta_file)
        assert idx_file.exists()
        assert meta_file.exists()

        new_store = VectorStore(dimension=EMBEDDING_DIM)
        new_store.load(idx_file, meta_file)

        assert new_store.is_initialized is True
        assert new_store.size == 150

    def test_missing_index_file_raises(self, tmp_path: Path) -> None:
        """Loading non-existent index file raises IndexNotFoundError."""
        store = VectorStore(dimension=EMBEDDING_DIM)
        with pytest.raises(IndexNotFoundError, match="index file not found"):
            store.load(tmp_path / "nonexistent.faiss", tmp_path / "meta.json")

    def test_uninitialized_search_raises(self) -> None:
        """Searching uninitialized store raises VectorStoreError."""
        store = VectorStore(dimension=EMBEDDING_DIM)
        query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        with pytest.raises(VectorStoreError, match="not initialized"):
            store.search(query, top_k=5)


class TestDenseRetrievalService:
    """Unit tests for DenseRetrievalService."""

    @pytest.fixture
    def mock_service(self, one_fifty_chunks: List[Chunk]) -> DenseRetrievalService:
        """Service initialized with an in-memory VectorStore."""
        embedder = EmbeddingService()
        store = VectorStore(dimension=EMBEDDING_DIM)

        embeddings = embedder.embed_chunks(one_fifty_chunks[:30])
        store.build(embeddings, one_fifty_chunks[:30])

        return DenseRetrievalService(embedding_service=embedder, vector_store=store)

    def test_retrieval_result_structure(self, mock_service: DenseRetrievalService) -> None:
        """retrieve() must return a RetrievalResponse containing RetrievalResult objects."""
        resp = mock_service.retrieve("Facilities operating SLA", top_k=3)

        assert isinstance(resp, RetrievalResponse)
        assert resp.query == "Facilities operating SLA"
        assert resp.top_k == 3
        assert len(resp.results) == 3

        first = resp.results[0]
        assert isinstance(first, RetrievalResult)
        assert first.rank == 1
        assert first.chunk_id != ""
        assert first.document_id != ""
        assert first.text != ""
        assert isinstance(first.similarity_score, float)

    def test_top_k_behavior(self, mock_service: DenseRetrievalService) -> None:
        """Retrieval must respect top_k parameter."""
        resp1 = mock_service.retrieve("Transport service hours", top_k=1)
        assert len(resp1.results) == 1

        resp5 = mock_service.retrieve("Transport service hours", top_k=5)
        assert len(resp5.results) == 5

    def test_similarity_score_ordering(self, mock_service: DenseRetrievalService) -> None:
        """Results must be ordered by similarity score descending."""
        resp = mock_service.retrieve("Security module incident escalation", top_k=5)
        scores = [r.similarity_score for r in resp.results]
        ranks = [r.rank for r in resp.results]

        assert scores == sorted(scores, reverse=True)
        assert ranks == list(range(1, len(scores) + 1))

    def test_empty_query_handling(self, mock_service: DenseRetrievalService) -> None:
        """Empty query must raise RetrievalError."""
        with pytest.raises(RetrievalError, match="empty"):
            mock_service.retrieve("")

        with pytest.raises(RetrievalError, match="empty"):
            mock_service.retrieve("   ")

    def test_invalid_top_k_handling(self, mock_service: DenseRetrievalService) -> None:
        """Out of bounds top_k raises RetrievalError."""
        with pytest.raises(RetrievalError, match="top_k"):
            mock_service.retrieve("Valid query", top_k=0)

        with pytest.raises(RetrievalError, match="top_k"):
            mock_service.retrieve("Valid query", top_k=999)

    def test_missing_index_handling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uninitialized service with missing index directory raises RetrievalError."""
        empty_store = VectorStore(dimension=EMBEDDING_DIM)
        monkeypatch.setattr(empty_store, "load", lambda *a, **kw: (_ for _ in ()).throw(IndexNotFoundError("Index file not found")))
        service = DenseRetrievalService(vector_store=empty_store)

        with pytest.raises(RetrievalError, match="Vector store is not initialized"):
            service.retrieve("Test query")
