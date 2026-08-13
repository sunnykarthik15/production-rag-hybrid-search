#!/usr/bin/env python3
"""
scripts/build_vector_index.py
==============================
CLI script to build and persist the FAISS dense vector index from processed chunks.

Usage
-----
    python scripts/build_vector_index.py

Exit codes
----------
    0 — Vector index successfully built and persisted.
    1 — Index building failed.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTOR_STORE_DIR,
)
from app.embeddings.service import EmbeddingService  # noqa: E402
from app.ingestion.loaders import load_chunks_jsonl  # noqa: E402
from app.retrieval.vector_store import VectorStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = _REPO_ROOT / "data" / "processed" / "chunks.jsonl"
SEP = "=" * 40


def main() -> int:
    print(f"\n{SEP}")
    print("BUILDING DENSE VECTOR INDEX")
    print(f"{SEP}\n")

    if not CHUNKS_PATH.exists():
        print(f"ERROR: Chunks file not found at: {CHUNKS_PATH}")
        return 1

    try:
        print(f"Loading chunks from {CHUNKS_PATH.relative_to(_REPO_ROOT)}...")
        chunks = list(load_chunks_jsonl(CHUNKS_PATH))
        print(f"Loaded {len(chunks)} chunks.")

        print(f"\nInitializing embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedder = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)

        print(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedder.embed_chunks(chunks)
        print(f"Generated embeddings array of shape {embeddings.shape}.")

        print(f"\nBuilding FAISS vector index (Cosine Similarity)...")
        store = VectorStore(dimension=embedder.expected_dim)
        store.build(embeddings, chunks)

        print(f"Persisting vector store to {VECTOR_STORE_DIR.relative_to(_REPO_ROOT)}...")
        store.save(index_path=FAISS_INDEX_PATH, metadata_path=METADATA_PATH)

        print(f"\n{SEP}")
        print(f"Indexed chunks: {store.size}")
        print(f"Index location: {FAISS_INDEX_PATH.relative_to(_REPO_ROOT)}")
        print(f"Metadata file:  {METADATA_PATH.relative_to(_REPO_ROOT)}")
        print(f"{SEP}")
        print("VECTOR INDEX BUILD PASSED")
        print(f"{SEP}\n")
        return 0

    except Exception as exc:
        logger.exception("Failed to build vector index: %s", exc)
        print(f"\nERROR: Index build failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
