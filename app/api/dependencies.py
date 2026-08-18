"""Dependency injection providers for Phase 8 FastAPI application.

Provides singleton-style lifecycle management for RAGService and
RerankedHybridService to avoid re-loading FAISS, BM25, and Cross-Encoder
models on every request.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from app.citations.service import CitationVerificationService
from app.generation.service import GenerationService, MockLLMProvider, OpenAILLMProvider
from app.rag.service import RAGService
from app.reranking.hybrid_reranker import RerankedHybridService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton service containers
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None
_hybrid_reranker: Optional[RerankedHybridService] = None
_initialized: bool = False


def get_rag_service() -> RAGService:
    """Return the singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_hybrid_reranker() -> RerankedHybridService:
    """Return the singleton RerankedHybridService instance.

    Reuses the hybrid reranker instance from the singleton RAGService
    to guarantee zero model or index duplication in memory.
    """
    global _hybrid_reranker
    if _hybrid_reranker is None:
        _hybrid_reranker = get_rag_service().hybrid_reranker
    return _hybrid_reranker


def initialize_services() -> None:
    """Initialize all singleton services (warm models on startup).

    Called once during application lifespan startup to ensure retrieval,
    reranking, and generation models are loaded before serving requests.
    """
    global _initialized

    if _initialized:
        logger.info("Services already initialized, skipping.")
        return

    rag_service = get_rag_service()

    try:
        rag_service.initialize()
        logger.info("RAGService initialization complete (retrieval + reranker warm).")
    except Exception as exc:
        logger.warning("RAGService initialization failed: %s", exc)

    _initialized = True
    logger.info("All Phase 8 API services initialized.")


def get_llm_provider_info() -> tuple[str, bool]:
    """Return current LLM provider name and whether it is a live external provider.

    Returns
    -------
    tuple[str, bool]
        (provider_name, is_live_llm) — e.g. ("mock", False) or ("openai", True).
    """
    from app.config import LLM_PROVIDER

    provider = LLM_PROVIDER.lower()
    is_live = provider == "openai" and bool(os.getenv("OPENAI_API_KEY"))
    return (provider if not is_live else "openai", is_live)


def check_component_health() -> dict[str, bool]:
    """Check readiness of individual components based on actual initialization states.

    Returns
    -------
    dict[str, bool]
        Component name → readiness boolean.
    """
    components = {
        "retrieval": False,
        "reranker": False,
        "generation": False,
        "citation_verifier": False,
    }

    try:
        rag = get_rag_service()

        # Check retrieval layer (VectorStore and BM25Store must be initialized)
        if rag.hybrid_reranker is not None and rag.hybrid_reranker.hybrid_service is not None:
            hs = rag.hybrid_reranker.hybrid_service
            vec_ready = getattr(getattr(hs, "vector_service", None), "vector_store", None)
            bm25_ready = getattr(getattr(hs, "bm25_service", None), "bm25_store", None)
            if (
                vec_ready is not None
                and vec_ready.is_initialized
                and bm25_ready is not None
                and bm25_ready.is_initialized
            ):
                components["retrieval"] = True

        # Check reranker layer (CrossEncoder must be loaded)
        if rag.hybrid_reranker is not None and rag.hybrid_reranker.reranker_service is not None:
            if rag.hybrid_reranker.reranker_service.is_initialized:
                components["reranker"] = True

        # Check generation provider availability
        if rag.generation_service is not None and rag.generation_service.provider is not None:
            components["generation"] = True

        # Check citation verifier readiness
        if rag.citation_service is not None and rag.citation_service.verifier is not None:
            components["citation_verifier"] = True

    except Exception as exc:
        logger.warning("Health check encountered error: %s", exc)

    return components


def reset_services() -> None:
    """Reset singleton services (for testing only)."""
    global _rag_service, _hybrid_reranker, _initialized
    _rag_service = None
    _hybrid_reranker = None
    _initialized = False
