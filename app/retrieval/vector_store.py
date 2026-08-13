"""
FAISS-backed local vector store.

Uses `faiss.IndexFlatIP` (Inner Product) with L2-normalized embeddings.
Because vectors are L2-normalized, Inner Product is strictly equivalent
to Cosine Similarity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np

from app.config import (
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTOR_STORE_DIR,
)
from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class IndexNotFoundError(VectorStoreError):
    """Raised when vector index or metadata files are missing from disk."""


class MetadataLoadError(VectorStoreError):
    """Raised when stored metadata file is corrupt or invalid."""


# ---------------------------------------------------------------------------
# VectorStore Implementation
# ---------------------------------------------------------------------------


class VectorStore:
    """FAISS-based local vector index manager.

    Maintains a 1-to-1 positional correspondence between FAISS index IDs
    (0, 1, ..., N-1) and chunk metadata records stored in a parallel JSON list.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: List[Dict[str, Any]] = []

    @property
    def is_initialized(self) -> bool:
        """Return True if index is loaded/built and metadata is populated."""
        return self.index is not None and len(self.metadata) == self.index.ntotal

    @property
    def size(self) -> int:
        """Return the number of vectors in the index."""
        return self.index.ntotal if self.index is not None else 0

    def build(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[Chunk],
    ) -> None:
        """Build the FAISS index from embeddings and corresponding Chunk objects.

        Parameters
        ----------
        embeddings : np.ndarray
            2D float32 numpy array of shape (N, dimension). Must be L2-normalized.
        chunks : Sequence[Chunk]
            Sequence of N Chunk objects matching embeddings row-by-row.

        Raises
        ------
        VectorStoreError
            If inputs are invalid or dimensions mismatch.
        """
        if len(embeddings) != len(chunks):
            raise VectorStoreError(
                f"Count mismatch between embeddings ({len(embeddings)}) and chunks ({len(chunks)})."
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embeddings shape {embeddings.shape} incompatible with dimension {self.dimension}."
            )

        logger.info("Building FAISS IndexFlatIP for %d vectors (dim=%d)...", len(chunks), self.dimension)

        # Create IndexFlatIP (Inner Product on L2-normalized vectors = Cosine Similarity)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)

        self.index = index
        self.metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "title": chunk.title,
                "department": chunk.department,
                "chunk_index": chunk.chunk_index,
                "source": chunk.source,
            }
            for chunk in chunks
        ]

        logger.info("Vector index built successfully with %d entries.", self.size)

    def save(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        """Persist the FAISS index and metadata to disk.

        Parameters
        ----------
        index_path : Path
            File path for the .faiss index file.
        metadata_path : Path
            File path for the metadata JSON file.

        Raises
        ------
        VectorStoreError
            If index is not built prior to saving.
        """
        if not self.is_initialized or self.index is None:
            raise VectorStoreError("Cannot save uninitialized vector store. Call build() or load() first.")

        # Ensure parent directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving FAISS index to %s...", index_path)
        faiss.write_index(self.index, str(index_path))

        logger.info("Saving metadata mapping (%d entries) to %s...", len(self.metadata), metadata_path)
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2)

        logger.info("Vector store persisted successfully.")

    def load(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        """Load FAISS index and metadata mapping from disk.

        Parameters
        ----------
        index_path : Path
            File path for the .faiss index file.
        metadata_path : Path
            File path for the metadata JSON file.

        Raises
        ------
        IndexNotFoundError
            If index file or metadata file does not exist.
        MetadataLoadError
            If metadata is corrupt or count mismatches index size.
        """
        if not index_path.exists():
            raise IndexNotFoundError(f"FAISS index file not found at: {index_path}")
        if not metadata_path.exists():
            raise IndexNotFoundError(f"Metadata file not found at: {metadata_path}")

        logger.info("Loading FAISS index from %s...", index_path)
        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"Failed to load FAISS index: {exc}") from exc

        if index.d != self.dimension:
            raise VectorStoreError(
                f"Loaded FAISS index dimension {index.d} does not match configured dimension {self.dimension}."
            )

        logger.info("Loading metadata mapping from %s...", metadata_path)
        try:
            with metadata_path.open(encoding="utf-8") as fh:
                metadata = json.load(fh)
        except Exception as exc:
            raise MetadataLoadError(f"Failed to load metadata JSON: {exc}") from exc

        if not isinstance(metadata, list) or len(metadata) != index.ntotal:
            raise MetadataLoadError(
                f"Metadata entry count ({len(metadata) if isinstance(metadata, list) else 0}) "
                f"does not match FAISS index vector count ({index.ntotal})."
            )

        self.index = index
        self.metadata = metadata
        logger.info("Vector store loaded successfully. %d vectors ready.", self.size)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        """Search top_k nearest neighbors using Cosine Similarity.

        Parameters
        ----------
        query_vector : np.ndarray
            1D or 2D L2-normalized float32 numpy array.
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        List[Tuple[Dict[str, Any], float, int]]
            List of tuples: (metadata_dict, similarity_score, 1_based_rank).

        Raises
        ------
        VectorStoreError
            If vector store is not initialized or input vector is invalid.
        """
        if not self.is_initialized or self.index is None:
            raise VectorStoreError("Vector store not initialized. Load or build index before search.")

        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}.")

        # Cap top_k to index size if index is smaller
        actual_k = min(top_k, self.size)

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        if query_vector.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query vector dimension {query_vector.shape[1]} != index dimension {self.dimension}."
            )

        # Execute FAISS search
        scores, indices = self.index.search(query_vector, actual_k)

        results: List[Tuple[Dict[str, Any], float, int]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item_metadata = self.metadata[idx]
            # Convert float32 score to Python float
            results.append((item_metadata, float(score), rank))

        return results
