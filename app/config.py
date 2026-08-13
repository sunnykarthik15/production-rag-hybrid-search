"""
Configuration for Phase 2: Dense Embeddings and Vector Retrieval.

All tunable parameters live here. Nothing is hardcoded in the service modules.
Paths are always resolved relative to the repository root so this works
regardless of the working directory.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — the parent of this file's package hierarchy
# ---------------------------------------------------------------------------
# app/config.py → repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

#: Hugging Face model name for sentence-transformers.
#: all-MiniLM-L6-v2 produces 384-dimensional embeddings, is CPU-friendly,
#: and is one of the most widely validated small retrieval models.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Expected output dimension of the embedding model above.
#: Checked at startup to detect model/index dimension mismatches early.
EMBEDDING_DIM: int = 384

#: Number of texts to embed in a single forward pass.
#: 32 is a safe default for CPU inference on a 150-chunk corpus.
EMBEDDING_BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Vector store paths  (generated/ignored — not committed to Git)
# ---------------------------------------------------------------------------

#: Root directory for all generated vector store files.
VECTOR_STORE_DIR: Path = _REPO_ROOT / "data" / "vector_store"

#: Path to the serialised FAISS index file.
FAISS_INDEX_PATH: Path = VECTOR_STORE_DIR / "index.faiss"

#: Path to the chunk-ID → metadata JSON mapping.
METADATA_PATH: Path = VECTOR_STORE_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

#: Default number of results returned by the dense retrieval service.
DEFAULT_TOP_K: int = 10

#: Absolute minimum value for top_k (guards against nonsensical values).
MIN_TOP_K: int = 1

#: Absolute maximum value for top_k (prevents accidental full-corpus scans).
MAX_TOP_K: int = 150

#: Default Reciprocal Rank Fusion (RRF) smoothing parameter.
DEFAULT_RRF_K: int = 60

#: Default candidate retrieval depth for hybrid retrieval components before RRF.
DEFAULT_HYBRID_CANDIDATE_K: int = 50

#: Hugging Face cross-encoder model name for Phase 5 reranking.
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Default number of hybrid candidate chunks to fetch before reranking.
DEFAULT_RERANK_CANDIDATE_K: int = 30


# ---------------------------------------------------------------------------
# Phase 6: LLM Generation & RAG Orchestration
# ---------------------------------------------------------------------------

#: LLM provider identifier ("mock", "openai").
LLM_PROVIDER: str = "mock"

#: Model name for generation provider.
LLM_MODEL_NAME: str = "gpt-3.5-turbo"

#: Sampling temperature for LLM generation (0.0 for deterministic output).
LLM_TEMPERATURE: float = 0.0

#: Maximum tokens allowed in generated output completion.
LLM_MAX_TOKENS: int = 512

#: Minimum Cross-Encoder reranker relevance score threshold for evidence sufficiency.
#: Scores below this threshold trigger an insufficient-information response.
RAG_MIN_RELEVANCE_SCORE: float = 0.0


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

#: Directory where evaluation result JSON files are written.
RESULTS_DIR: Path = _REPO_ROOT / "results"
