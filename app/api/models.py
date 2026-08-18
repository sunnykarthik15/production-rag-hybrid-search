"""Pydantic request/response models for Phase 8 FastAPI endpoints."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.config import API_MAX_QUERY_LENGTH, DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for POST /query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=API_MAX_QUERY_LENGTH,
        description="User question or search query.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="Number of evidence chunks to retrieve.",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string must not be empty or whitespace-only.")
        return stripped


class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve debug endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=API_MAX_QUERY_LENGTH,
        description="User question or search query.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=MIN_TOP_K,
        le=MAX_TOP_K,
        description="Number of reranked chunks to return.",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query string must not be empty or whitespace-only.")
        return stripped


# ---------------------------------------------------------------------------
# Response Sub-Models
# ---------------------------------------------------------------------------


class LatencyMetadata(BaseModel):
    """Monotonic timing metadata for API responses."""

    retrieval_latency_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Time spent in retrieval+reranking (ms)."
    )
    generation_latency_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Time spent in LLM generation (ms)."
    )
    verification_latency_ms: Optional[float] = Field(
        default=None, ge=0.0, description="Time spent in citation verification (ms)."
    )
    total_latency_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock latency of the request (ms)."
    )
    llm_provider: str = Field(
        ..., description="LLM provider used ('mock' or 'openai')."
    )
    is_live_llm: bool = Field(
        ..., description="True if a live external LLM API was invoked."
    )
    query_hash: Optional[str] = Field(
        default=None, description="Deterministic SHA-256 hash of user query."
    )
    request_id: Optional[str] = Field(
        default=None, description="Unique correlation ID for the request."
    )


class SourceItem(BaseModel):
    """Serialised evidence source from RAG retrieval results."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Parent document identifier.")
    title: str = Field(..., description="Parent document title.")
    department: str = Field(..., description="Associated department.")
    rank: int = Field(..., ge=1, description="1-based rank in search results.")
    score: float = Field(..., description="Reranker relevance score.")
    text: Optional[str] = Field(default=None, description="Raw chunk text content.")


class RetrieveResultItem(BaseModel):
    """Single retrieval result item for the /retrieve debug endpoint."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="Parent document identifier.")
    title: str = Field(..., description="Parent document title.")
    department: str = Field(..., description="Associated department.")
    rank: int = Field(..., ge=1, description="1-based rank in search results.")
    score: float = Field(..., description="Similarity/relevance score.")
    text: str = Field(..., description="Chunk text content.")


class ComponentStatus(BaseModel):
    """Component-level readiness flags for the health/readiness endpoints."""

    retrieval: bool = Field(..., description="Hybrid retrieval service readiness.")
    reranker: bool = Field(..., description="Cross-encoder reranker readiness.")
    generation: bool = Field(..., description="LLM generation service readiness.")
    citation_verifier: bool = Field(
        ..., description="Citation verification service readiness."
    )


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health liveness endpoint."""

    status: str = Field(
        ..., description="Overall application liveness status ('healthy', 'alive', or 'degraded')."
    )
    environment: str = Field(
        ..., description="LLM environment mode ('mock' or 'live')."
    )
    components: Optional[ComponentStatus] = Field(
        default=None, description="Optional component status flags."
    )


class ReadyResponse(BaseModel):
    """Response body for GET /ready readiness endpoint."""

    status: str = Field(
        ..., description="Readiness status ('ready' or 'not_ready')."
    )
    ready: bool = Field(..., description="Boolean readiness flag.")
    components: ComponentStatus = Field(
        ..., description="Individual component readiness flags."
    )
    timestamp: str = Field(..., description="ISO UTC timestamp.")


class QueryResponse(BaseModel):
    """Response body for POST /query endpoint."""

    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Grounded natural language answer.")
    sources: List[SourceItem] = Field(
        default_factory=list, description="Evidence source citations."
    )
    verification: Optional[Dict] = Field(
        default=None, description="Citation verification and groundedness analysis."
    )
    groundedness_score: Optional[float] = Field(
        default=None, description="Overall groundedness score (0.0 to 1.0)."
    )
    metadata: LatencyMetadata = Field(
        ..., description="Request timing and provider metadata."
    )


class RetrieveResponse(BaseModel):
    """Response body for POST /retrieve debug endpoint."""

    query: str = Field(..., description="Original user query.")
    top_k: int = Field(..., description="Requested top_k count.")
    results: List[RetrieveResultItem] = Field(
        default_factory=list, description="Ordered list of reranked retrieval results."
    )


# ---------------------------------------------------------------------------
# Error Models
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Structured error information envelope."""

    code: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[str] = Field(
        default=None, description="Optional additional diagnostic context."
    )


class ErrorResponse(BaseModel):
    """Standardised API error response wrapper."""

    error: ErrorDetail = Field(..., description="Error details envelope.")
