"""FastAPI route definitions for Phase 8 & 9 Production API.

Implements:
- GET  /health   — Liveness probe (application process is responsive).
- GET  /ready    — Readiness probe (all models, indices, and dependencies loaded).
- GET  /metrics  — In-memory operational metrics snapshot.
- POST /query    — Full RAG pipeline (retrieval → generation → verification).
- POST /retrieve — Debug/inspection retrieval-only endpoint.

All endpoints delegate to existing RAGService and RerankedHybridService
without duplicating any retrieval, generation, or verification logic.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    check_component_health,
    get_hybrid_reranker,
    get_llm_provider_info,
    get_rag_service,
)
from app.api.models import (
    ComponentStatus,
    HealthResponse,
    LatencyMetadata,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResultItem,
    SourceItem,
)
from app.observability.logging import get_query_hash
from app.observability.metrics import get_metrics_collector

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return application liveness status and component readiness.

    Indicates whether the API process is alive and responsive.
    """
    provider_name, is_live = get_llm_provider_info()
    component_flags = check_component_health()
    all_healthy = all(component_flags.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment="live" if is_live else "mock",
        components=ComponentStatus(**component_flags),
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready_check() -> JSONResponse:
    """Return application readiness status.

    Verifies whether underlying dependencies (VectorStore, BM25Store, CrossEncoder)
    are initialized and capable of serving search and generation requests.
    Returns HTTP 200 when ready, HTTP 503 when dependencies are uninitialized.
    """
    component_flags = check_component_health()
    all_ready = all(component_flags.values())
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    response_payload = ReadyResponse(
        status="ready" if all_ready else "not_ready",
        ready=all_ready,
        components=ComponentStatus(**component_flags),
        timestamp=now_iso,
    )

    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content=response_payload.model_dump(mode="json"),
    )


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Return in-memory operational metrics snapshot.

    Never exposes secrets, API keys, filesystem paths, or raw query text.
    """
    metrics = get_metrics_collector()
    return metrics.get_snapshot()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request_body: QueryRequest, http_request: Request) -> QueryResponse:
    """Execute full RAG pipeline: retrieval → generation → citation verification.

    Delegates entirely to existing RAGService.answer() without duplicating
    any retrieval or generation logic.
    """
    provider_name, is_live = get_llm_provider_info()
    rag_service = get_rag_service()
    metrics = get_metrics_collector()

    query_text = request_body.query
    query_hash = get_query_hash(query_text)
    request_id = getattr(http_request.state, "request_id", None)

    total_start = time.monotonic()

    # Execute the full RAG pipeline
    rag_response = rag_service.answer(query=query_text, top_k=request_body.top_k)

    total_end = time.monotonic()
    total_latency_ms = (total_end - total_start) * 1000.0

    # Record RAG-specific operational metrics
    metrics.increment("retrieval_executions")
    if rag_response.sources:
        metrics.increment("evidence_sufficient")
        metrics.increment("llm_executions")
    else:
        metrics.increment("evidence_insufficient")
        metrics.increment("llm_skipped_due_to_insufficient_evidence")

    if rag_response.verification is not None:
        metrics.increment("citation_verifications")
        if rag_response.verification.is_fully_grounded:
            metrics.increment("grounded_answers")
        else:
            metrics.increment("ungrounded_answers")

    # Map RAGSource objects to API SourceItem models
    sources = [
        SourceItem(
            chunk_id=src.chunk_id,
            document_id=src.document_id,
            title=src.title,
            department=src.department,
            rank=src.rank,
            score=src.score,
            text=src.text,
        )
        for src in rag_response.sources
    ]

    # Serialise verification result if present
    verification_dict = None
    if rag_response.verification is not None:
        verification_dict = rag_response.verification.model_dump(mode="json")

    metadata = LatencyMetadata(
        retrieval_latency_ms=None,
        generation_latency_ms=None,
        verification_latency_ms=None,
        total_latency_ms=round(total_latency_ms, 2),
        llm_provider=provider_name,
        is_live_llm=is_live,
        query_hash=query_hash,
        request_id=request_id,
    )

    return QueryResponse(
        query=rag_response.query,
        answer=rag_response.answer,
        sources=sources,
        verification=verification_dict,
        groundedness_score=rag_response.groundedness_score,
        metadata=metadata,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_only(request_body: RetrieveRequest) -> RetrieveResponse:
    """Execute reranked hybrid retrieval WITHOUT generation or citation verification.

    Useful for debugging and inspecting retrieval quality independently.
    """
    hybrid_reranker = get_hybrid_reranker()
    metrics = get_metrics_collector()

    start = time.monotonic()
    response = hybrid_reranker.retrieve(
        query=request_body.query,
        top_k=request_body.top_k,
    )
    duration_ms = (time.monotonic() - start) * 1000.0

    metrics.increment("retrieval_executions")
    metrics.record_latency("retrieval", duration_ms)

    results = [
        RetrieveResultItem(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            title=item.title,
            department=item.department,
            rank=item.rank,
            score=item.similarity_score,
            text=item.text,
        )
        for item in response.results
    ]

    return RetrieveResponse(
        query=response.query,
        top_k=response.top_k,
        results=results,
    )
