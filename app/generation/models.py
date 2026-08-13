"""Pydantic data models for the LLM generation layer."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """Container for input query and context chunks passed to the LLM generation service."""

    query: str = Field(..., description="Original natural language user query.")
    context: List[str] = Field(
        default_factory=list,
        description="Formatted evidence text blocks for grounded generation.",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string must not be empty.")
        return v.strip()


class GenerationResponse(BaseModel):
    """Container for generated answer and optional metadata from the LLM provider."""

    answer: str = Field(..., description="Grounded natural language answer text.")
    model_name: str = Field(default="mock", description="Name of the LLM model used.")
    prompt_tokens: Optional[int] = Field(
        default=None, description="Number of tokens in prompt, if reported."
    )
    completion_tokens: Optional[int] = Field(
        default=None, description="Number of tokens in completion, if reported."
    )

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Answer text must not be empty.")
        return v.strip()
