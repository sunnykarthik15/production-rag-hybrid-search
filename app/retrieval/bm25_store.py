"""BM25 Lexical Store Implementation.

Indexes document text chunks using BM25Okapi for keyword/lexical search.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Sequence, Tuple, Union

from rank_bm25 import BM25Okapi

from app.ingestion.models import Chunk

logger = logging.getLogger(__name__)

# Tokenization regex matching words, numbers, and hyphenated terms (e.g., NX-FAC-100)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Deterministic tokenization for BM25 lexical search.

    Converts text to lowercase and extracts alphanumeric words, numbers,
    and hyphenated identifiers (e.g., 'NX-FAC-100' or 'DOC001').

    Parameters
    ----------
    query / text : str
        Input string to tokenize.

    Returns
    -------
    List[str]
        Ordered list of lowercase token strings.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BM25StoreError(Exception):
    """Exception raised for errors in BM25Store operations."""


class BM25Store:
    """Store managing BM25 lexical index construction and searching.

    Maps indexed corpus positions to original chunk metadata.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks_metadata: List[Dict[str, Any]] = []
        self._is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Return True if index is built and ready for search."""
        return self._is_initialized

    @property
    def chunk_count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._chunks_metadata)

    def build(self, chunks: Sequence[Union[Chunk, Dict[str, Any]]]) -> None:
        """Build BM25 index from input list of Chunk objects or dicts.

        Parameters
        ----------
        chunks : Sequence[Chunk | Dict[str, Any]]
            Sequence of chunk objects or dictionary records.

        Raises
        ------
        BM25StoreError
            If input chunks sequence is empty.
        """
        if not chunks:
            raise BM25StoreError("Cannot build BM25 index from empty chunks list.")

        metadata_list: List[Dict[str, Any]] = []
        corpus_tokens: List[List[str]] = []

        for chunk in chunks:
            if isinstance(chunk, Chunk):
                c_dict = chunk.model_dump()
            else:
                c_dict = dict(chunk)

            # Include title and text for lexical term matching
            full_text = f"{c_dict.get('title', '')} {c_dict.get('text', '')}"
            tokens = tokenize(full_text)

            corpus_tokens.append(tokens)
            metadata_list.append(c_dict)

        self._bm25 = BM25Okapi(corpus_tokens)
        self._chunks_metadata = metadata_list
        self._is_initialized = True
        logger.info(f"Successfully built BM25 index for {len(metadata_list)} chunks.")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float, int]]:
        """Search BM25 index with input query string.

        Parameters
        ----------
        query : str
            Natural language query text.
        top_k : int
            Number of top results to return.

        Returns
        -------
        List[Tuple[Dict[str, Any], float, int]]
            List of (chunk_metadata, score, 1-based rank) tuples sorted by score descending.

        Raises
        ------
        BM25StoreError
            If index is not built prior to search.
        """
        if not self._is_initialized or self._bm25 is None:
            raise BM25StoreError("BM25Store is not initialized. Call build() first.")

        if top_k < 1:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []

        # Sort indices by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[Tuple[Dict[str, Any], float, int]] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            score = float(scores[idx])
            meta = self._chunks_metadata[idx]
            results.append((meta, score, rank))

        return results
