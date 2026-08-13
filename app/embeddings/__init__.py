"""Embeddings package for dense vector representation."""

from app.embeddings.service import (
    DimensionMismatchError,
    EmbeddingError,
    EmbeddingService,
    ModelLoadError,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingError",
    "ModelLoadError",
    "DimensionMismatchError",
]
