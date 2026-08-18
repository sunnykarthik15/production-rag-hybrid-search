"""Phase 7: Citation Verification, Answer Groundedness, and Attribution module."""

from __future__ import annotations

from app.citations.extractor import extract_claims, parse_citations_from_text, split_sentences
from app.citations.models import (
    AnswerType,
    CitationVerificationResult,
    Claim,
    ClaimVerification,
    UnsupportedReason,
)
from app.citations.service import CitationVerificationError, CitationVerificationService
from app.citations.verifier import ClaimVerifier

__all__ = [
    "AnswerType",
    "UnsupportedReason",
    "Claim",
    "ClaimVerification",
    "CitationVerificationResult",
    "ClaimVerifier",
    "CitationVerificationService",
    "CitationVerificationError",
    "extract_claims",
    "parse_citations_from_text",
    "split_sentences",
]
