"""BM25 Retrieval Service.

Wraps BM25Store to provide a clean service interface returning structured RetrievalResponse objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

from app.config import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K
from app.ingestion.loaders import load_chunks_jsonl
from app.ingestion.models import Chunk
from app.retrieval.bm25_store import BM25Store, BM25StoreError
from app.retrieval.models import RetrievalResponse, RetrievalResult
from app.retrieval.service import RetrievalError

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"


class BM25RetrievalService:
    """Service providing lexical BM25 retrieval against document chunks.

    Constructs or uses a BM25Store to process queries and output
    structured RetrievalResponse models compatible with the dense retrieval pipeline.
    """

    def __init__(self, bm25_store: BM25Store | None = None) -> None:
        self.bm25_store = bm25_store or BM25Store()

    def initialize(
        self,
        chunks: Sequence[Union[Chunk, dict]] | None = None,
        chunks_path: Path | str | None = None,
    ) -> None:
        """Initialize BM25Store by loading and indexing chunks.

        Parameters
        ----------
        chunks : Sequence[Chunk | dict], optional
            Direct sequence of chunk objects/dicts.
        chunks_path : Path | str, optional
            File path to chunks JSONL file. Defaults to data/processed/chunks.jsonl.

        Raises
        ------
        RetrievalError
            If chunk loading or index construction fails.
        """
        try:
            if chunks is not None:
                chunk_list = list(chunks)
            else:
                target_path = Path(chunks_path) if chunks_path else _DEFAULT_CHUNKS_PATH
                if not target_path.exists():
                    raise RetrievalError(f"Chunks file not found for BM25 initialization: {target_path}")
                chunk_list = list(load_chunks_jsonl(target_path))

            self.bm25_store.build(chunk_list)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Failed to initialize BM25 retrieval service: {exc}") from exc

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RetrievalResponse:
        """Search BM25 index for top_k lexical matches to the input query string.

        Parameters
        ----------
        query : str
            Input natural language query text.
        top_k : int
            Number of top results to retrieve (between 1 and 150).

        Returns
        -------
        RetrievalResponse
            Structured container with query, top_k, and ordered RetrievalResult list.

        Raises
        ------
        RetrievalError
            If query is empty, top_k is invalid, or index loading fails.
        """
        if not query or not query.strip():
            raise RetrievalError("Query string must not be empty.")

        if not isinstance(top_k, int) or top_k < MIN_TOP_K or top_k > MAX_TOP_K:
            raise RetrievalError(
                f"top_k must be an integer between {MIN_TOP_K} and {MAX_TOP_K}, got {top_k}."
            )

        if not self.bm25_store.is_initialized:
            try:
                self.initialize()
            except Exception as exc:
                raise RetrievalError(
                    f"BM25 store is not initialized and auto-initialization failed: {exc}"
                ) from exc

        try:
            raw_results = self.bm25_store.search(query=query, top_k=top_k)
        except BM25StoreError as exc:
            raise RetrievalError(f"BM25 search failed: {exc}") from exc

        formatted_results = [
            RetrievalResult(
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                text=meta["text"],
                title=meta["title"],
                department=meta["department"],
                similarity_score=score,
                rank=rank,
            )
            for meta, score, rank in raw_results
        ]

        return RetrievalResponse(
            query=query.strip(),
            results=formatted_results,
            top_k=top_k,
        )

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResponse:
        """Alias for search() to align with DenseRetrievalService interface."""
        return self.search(query=query, top_k=top_k)
