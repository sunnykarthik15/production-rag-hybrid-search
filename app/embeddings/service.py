"""
Dense Embedding Service using sentence-transformers.

Provides local, CPU-compatible, deterministic embedding generation
using `sentence-transformers/all-MiniLM-L6-v2`.
"""

from __future__ import annotations

import logging
from typing import List, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
)
from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base exception for embedding operations."""


class ModelLoadError(EmbeddingError):
    """Raised when the sentence-transformer model fails to load."""


class DimensionMismatchError(EmbeddingError):
    """Raised when output embedding dimension does not match expected dimension."""


# ---------------------------------------------------------------------------
# Service Implementation
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Service for generating normalized dense embeddings.

    Loads the SentenceTransformer model once upon initialization and provides
    methods for embedding single queries, lists of text strings, or Chunk objects.
    All embeddings returned are L2-normalized float32 numpy arrays.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        expected_dim: int = EMBEDDING_DIM,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.device = device
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the model instance if not already loaded."""
        if self._model is None:
            logger.info("Loading embedding model '%s' on %s...", self.model_name, self.device)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load sentence-transformer model '{self.model_name}': {exc}"
                ) from exc

            # Verify model output dimension
            dim = self._model.get_sentence_embedding_dimension()
            if dim != self.expected_dim:
                raise DimensionMismatchError(
                    f"Model '{self.model_name}' produces {dim}-dim embeddings, "
                    f"expected {self.expected_dim}."
                )
            logger.info("Model loaded successfully. Embedding dimension: %d", dim)

        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a single L2-normalized 1D embedding array for a text string.

        Parameters
        ----------
        text : str
            The input string to embed. Must be non-empty.

        Returns
        -------
        np.ndarray
            1D float32 array of shape (expected_dim,).

        Raises
        ------
        EmbeddingError
            If text is empty or embedding fails.
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot generate embedding for an empty string.")

        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> np.ndarray:
        """Generate L2-normalized 2D embeddings for a sequence of text strings.

        Parameters
        ----------
        texts : Sequence[str]
            List of input strings.
        batch_size : int
            Number of items to embed in a single batch pass.

        Returns
        -------
        np.ndarray
            2D float32 array of shape (N, expected_dim).

        Raises
        ------
        EmbeddingError
            If texts sequence is empty or invalid.
        """
        if not texts:
            raise EmbeddingError("Cannot generate embeddings for an empty text list.")

        model = self._get_model()

        try:
            embeddings = model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Applies L2 normalization
            )
        except Exception as exc:
            raise EmbeddingError(f"Embedding encoding failed: {exc}") from exc

        embeddings = embeddings.astype(np.float32)

        # Validate shape
        if embeddings.ndim != 2 or embeddings.shape[1] != self.expected_dim:
            raise DimensionMismatchError(
                f"Expected embeddings shape (N, {self.expected_dim}), got {embeddings.shape}"
            )

        return embeddings

    def embed_chunks(
        self,
        chunks: Sequence[Chunk],
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> np.ndarray:
        """Generate embeddings for a list of ingestion Chunk objects.

        Parameters
        ----------
        chunks : Sequence[Chunk]
            List of Chunk objects.
        batch_size : int
            Batch size for embedding generation.

        Returns
        -------
        np.ndarray
            2D float32 array of shape (len(chunks), expected_dim).
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)
