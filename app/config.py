"""
Configuration for Phase 2: Dense Embeddings and Vector Retrieval.

All tunable parameters live here. Nothing is hardcoded in the service modules.
Paths are always resolved relative to the repository root so this works
regardless of the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

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
# Phase 7: Citation Verification & Answer Groundedness
# ---------------------------------------------------------------------------

#: Baseline minimum lexical/content-token overlap score required for claim support.
CITATION_MIN_OVERLAP_THRESHOLD: float = 0.40

#: Baseline entity/number/date match threshold (1.0 = strict 100% entity containment).
CITATION_ENTITY_MATCH_THRESHOLD: float = 1.0

#: Strict mode flag for citation verification warnings.
CITATION_STRICT_MODE: bool = False


# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------

#: Directory where evaluation result JSON files are written.
RESULTS_DIR: Path = _REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Phase 8 & 9: Production API, Observability & Deployment Configuration
# ---------------------------------------------------------------------------


def _get_env_int(key: str, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    """Safely parse integer environment variables with boundary checks and fallback."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        parsed = int(val.strip())
        if min_val is not None and parsed < min_val:
            return default
        if max_val is not None and parsed > max_val:
            return default
        return parsed
    except (ValueError, TypeError):
        return default


def _get_env_list(key: str, default: List[str]) -> List[str]:
    """Safely parse comma-separated string environment variables into a list."""
    val = os.getenv(key)
    if not val:
        return default
    items = [item.strip() for item in val.split(",") if item.strip()]
    return items if items else default


#: Host address for the FastAPI/Uvicorn server.
API_HOST: str = os.getenv("API_HOST", "0.0.0.0").strip()

#: Port for the FastAPI/Uvicorn server.
API_PORT: int = _get_env_int("API_PORT", default=8000, min_val=1, max_val=65535)

#: Application title shown in OpenAPI/Swagger documentation.
API_TITLE: str = os.getenv("API_TITLE", "Nexora RAG API").strip()

#: API version string.
API_VERSION: str = os.getenv("API_VERSION", "1.0.0").strip()

#: Maximum allowed query string length (characters).
API_MAX_QUERY_LENGTH: int = _get_env_int("API_MAX_QUERY_LENGTH", default=2000, min_val=1, max_val=100000)

#: Allowed CORS origins.
API_CORS_ORIGINS: List[str] = _get_env_list(
    "API_CORS_ORIGINS", default=["http://localhost:3000", "http://localhost:8000"]
)

#: Application logging level (e.g., DEBUG, INFO, WARNING, ERROR).
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper().strip()

#: Enable or disable in-memory metrics collection.
METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower().strip() in ("true", "1", "yes")

