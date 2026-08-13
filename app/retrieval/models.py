"""Data models for dense vector retrieval results."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class RetrievalResult(BaseModel):
    """Structured result item from vector retrieval."""

    chunk_id: str = Field(..., description="Unique chunk ID.")
    document_id: str = Field(..., description="ID of parent document.")
    text: str = Field(..., description="Chunk text content.")
    title: str = Field(..., description="Parent document title.")
    department: str = Field(..., description="Associated department.")
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score between query and chunk (range -1.0 to 1.0).",
    )
    rank: int = Field(..., ge=1, description="1-based rank position in search results.")

    @field_validator("chunk_id", "document_id", "text")
    @classmethod
    def string_fields_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be empty.")
        return v


class RetrievalResponse(BaseModel):
    """Structured response container for query retrieval."""

    query: str = Field(..., description="Original input query.")
    results: List[RetrievalResult] = Field(
        default_factory=list,
        description="List of retrieved chunks ordered by similarity score descending.",
    )
    top_k: int = Field(..., ge=1, description="Requested top_k result count.")
