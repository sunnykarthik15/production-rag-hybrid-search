"""Pydantic data models for RAG orchestration requests, citations, and responses."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator

from app.config import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K


class RAGSource(BaseModel):
    """Clean citation metadata for a single evidence chunk used in RAG generation."""

    chunk_id: str = Field(..., description="Unique chunk ID.")
    document_id: str = Field(..., description="Parent document ID.")
    title: str = Field(..., description="Parent document title.")
    department: str = Field(..., description="Associated department.")
    rank: int = Field(..., ge=1, description="1-based rank position in reranked search results.")
    score: float = Field(..., description="Relevance score assigned to chunk by reranker.")

    @field_validator("chunk_id", "document_id", "title")
    @classmethod
    def string_fields_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Citation field must not be empty.")
        return v.strip()


class RAGRequest(BaseModel):
    """Request container for end-to-end RAG question answering."""

    query: str = Field(..., description="User question or search query.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K, description="Number of evidence chunks to retrieve.")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string must not be empty.")
        return v.strip()


class RAGResponse(BaseModel):
    """End-to-end response container containing grounded answer and source citations."""

    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Grounded natural language answer.")
    sources: List[RAGSource] = Field(
        default_factory=list, description="Ordered list of evidence sources used for generation."
    )

    @field_validator("query", "answer")
    @classmethod
    def string_fields_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty.")
        return v.strip()
