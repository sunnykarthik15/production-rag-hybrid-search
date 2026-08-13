"""Unit tests for app.embeddings module."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import EMBEDDING_DIM
from app.embeddings.service import (
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingService,
)
from app.ingestion.models import Chunk


class TestEmbeddingService:
    """Unit test suite for EmbeddingService."""

    @pytest.fixture(scope="class")
    def embedder(self) -> EmbeddingService:
        """Shared EmbeddingService instance for unit tests."""
        return EmbeddingService()

    def test_embedding_output_shape(self, embedder: EmbeddingService) -> None:
        """Single and batch embeddings must match expected dimensions."""
        single_vec = embedder.embed_text("Sample text query")
        assert isinstance(single_vec, np.ndarray)
        assert single_vec.shape == (EMBEDDING_DIM,)

        batch_vecs = embedder.embed_texts(["Text one", "Text two", "Text three"])
        assert isinstance(batch_vecs, np.ndarray)
        assert batch_vecs.shape == (3, EMBEDDING_DIM)

    def test_deterministic_embedding_behavior(self, embedder: EmbeddingService) -> None:
        """Embedding the same text twice must yield identical vectors."""
        vec1 = embedder.embed_text("Nexora smart campus facilities module")
        vec2 = embedder.embed_text("Nexora smart campus facilities module")
        np.testing.assert_array_almost_equal(vec1, vec2, decimal=5)

    def test_normalization(self, embedder: EmbeddingService) -> None:
        """Returned embeddings must be L2-normalized (length == 1.0)."""
        vec = embedder.embed_text("Testing L2 normalization vector length")
        norm = np.linalg.norm(vec)
        assert pytest.approx(norm, abs=1e-5) == 1.0

        batch_vecs = embedder.embed_texts(["One", "Two", "Three"])
        batch_norms = np.linalg.norm(batch_vecs, axis=1)
        for n in batch_norms:
            assert pytest.approx(n, abs=1e-5) == 1.0

    def test_embed_chunks(self, embedder: EmbeddingService, valid_chunk: Chunk) -> None:
        """embed_chunks must accept Chunk objects and return valid 2D array."""
        chunks = [valid_chunk, valid_chunk]
        vecs = embedder.embed_chunks(chunks)
        assert vecs.shape == (2, EMBEDDING_DIM)

    def test_empty_query_handling(self, embedder: EmbeddingService) -> None:
        """Empty or whitespace-only query text must raise EmbeddingError."""
        with pytest.raises(EmbeddingError, match="empty string"):
            embedder.embed_text("")

        with pytest.raises(EmbeddingError, match="empty string"):
            embedder.embed_text("    ")

        with pytest.raises(EmbeddingError, match="empty text list"):
            embedder.embed_texts([])

    def test_dimension_mismatch_validation(self) -> None:
        """Configuring wrong expected_dim must raise DimensionMismatchError."""
        bad_embedder = EmbeddingService(expected_dim=512)
        with pytest.raises(DimensionMismatchError, match="expected 512"):
            bad_embedder.embed_text("Test")
