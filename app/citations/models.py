"""Pydantic data models for Phase 7 Citation Verification, Answer Groundedness, and Attribution."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AnswerType(str, Enum):
    """Categorical classification of generated answer response type."""

    GROUNDED_ANSWER = "grounded_answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FABRICATED_ANSWER = "fabricated_answer"
    UNSUPPORTED_ANSWER = "unsupported_answer"


class UnsupportedReason(str, Enum):
    """Specific diagnostic reason why a claim or citation failed verification."""

    NONE = "none"
    ENTITY_MISMATCH = "entity_mismatch"
    NUMERICAL_MISMATCH = "numerical_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    CONTRADICTION = "contradiction"
    NO_SUPPORTING_SOURCE = "no_supporting_source"
    WRONG_CITATION = "wrong_citation"
    INVALID_CITATION_INDEX = "invalid_citation_index"
    MISSING_CITATION = "missing_citation"


class Claim(BaseModel):
    """Represents a single segmented factual assertion from an answer."""

    claim_id: str = Field(..., description="Unique claim identifier within response (e.g. 'C001').")
    text: str = Field(..., description="Clean text of factual claim without citation markers.")
    raw_text: str = Field(..., description="Original claim text containing inline citation markers.")
    citation_indices: List[int] = Field(
        default_factory=list,
        description="1-based source indices cited by this claim (e.g. [1, 2]).",
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Explicit chunk IDs cited by this claim, if present.",
    )
    is_factual: bool = Field(
        default=True,
        description="False for conversational pleasantries, headings, or boilerplate refusals.",
    )

    @field_validator("claim_id", "text", "raw_text")
    @classmethod
    def fields_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Claim string fields must not be empty.")
        return v.strip()


class ClaimVerification(BaseModel):
    """Granular verification results and diagnostic signals for a single claim."""

    claim: Claim = Field(..., description="The verified claim instance.")
    is_grounded: bool = Field(
        ...,
        description="True if factual claim is supported by retrieved evidence chunks.",
    )
    is_citation_correct: bool = Field(
        ...,
        description="True if cited source chunk(s) actually support the claim.",
    )
    has_citation: bool = Field(
        ...,
        description="True if the claim contains at least one inline citation reference.",
    )
    support_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence / content overlap score of supporting evidence (0.0 to 1.0).",
    )
    supporting_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs verified to provide evidentiary support for this claim.",
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk IDs referenced by the citation marker(s).",
    )
    unsupported_reason: UnsupportedReason = Field(
        default=UnsupportedReason.NONE,
        description="Diagnostic failure reason if ungrounded or incorrectly cited.",
    )
    signals: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw diagnostic sub-scores (entity_match, token_overlap, lcs_ratio, polarity).",
    )


class CitationVerificationResult(BaseModel):
    """Complete post-generation verification and groundedness analysis for a RAG response."""

    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Generated answer text.")
    answer_type: AnswerType = Field(
        default=AnswerType.GROUNDED_ANSWER,
        description="Categorical response type (grounded, insufficient, fabricated, unsupported).",
    )
    refusal_correct: Optional[bool] = Field(
        default=None,
        description="True if unanswerable query was faithfully answered with insufficient-evidence refusal.",
    )
    claims: List[ClaimVerification] = Field(
        default_factory=list,
        description="List of verified individual claims.",
    )

    # Claim Groundedness Metrics
    total_claims: int = Field(default=0, ge=0, description="Total number of factual claims evaluated.")
    grounded_claims_count: int = Field(
        default=0, ge=0, description="Number of factual claims supported by evidence."
    )
    ungrounded_claims_count: int = Field(
        default=0, ge=0, description="Number of factual claims lacking evidentiary support."
    )
    claim_groundedness_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of factual claims supported by evidence (0.0 to 1.0).",
    )
    is_fully_grounded: bool = Field(
        default=True,
        description="True if 100% of factual claims are verified as grounded.",
    )

    # Citation Correctness & Completeness Metrics
    total_citations: int = Field(
        default=0, ge=0, description="Total number of citation references in answer."
    )
    correct_citations_count: int = Field(
        default=0, ge=0, description="Number of citations that correctly support their claim."
    )
    citation_precision: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision of citations (correct citations / total citations). None when total_citations is 0.",
    )
    citation_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Completeness/recall of citations (cited grounded claims / grounded claims).",
    )
    citation_f1: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Harmonic mean of citation precision and citation recall. None when total_citations is 0.",
    )
    is_fully_cited: bool = Field(
        default=True,
        description="True if all grounded factual claims have appropriate citations.",
    )

    # Failure Mode Diagnostics Accounting
    spurious_citations_count: int = Field(
        default=0, ge=0, description="Citations attached to refusal or irrelevant content."
    )
    invalid_citations_count: int = Field(
        default=0, ge=0, description="Citations referencing out-of-range source indices."
    )
    hallucinated_claims_count: int = Field(
        default=0,
        ge=0,
        description="Claims introducing conflicting entities, dates, numbers, or contradictions.",
    )
    unsupported_claims_count: int = Field(
        default=0,
        ge=0,
        description="Claims without supporting evidence in retrieved context.",
    )
    wrong_citations_count: int = Field(
        default=0,
        ge=0,
        description="Claims citing a source that does not support the claim.",
    )
    missing_citations_count: int = Field(
        default=0,
        ge=0,
        description="Grounded claims that omit required source citation tags.",
    )

    @field_validator("query", "answer")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query and answer strings must not be empty.")
        return v.strip()
