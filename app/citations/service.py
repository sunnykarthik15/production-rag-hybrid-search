"""Citation Verification Service orchestrator."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from app.citations.extractor import (
    INSUFFICIENT_INFO_SENTINEL,
    extract_claims,
)
from app.citations.models import (
    AnswerType,
    CitationVerificationResult,
    ClaimVerification,
    UnsupportedReason,
)
from app.citations.verifier import ClaimVerifier
from app.config import (
    CITATION_ENTITY_MATCH_THRESHOLD,
    CITATION_MIN_OVERLAP_THRESHOLD,
)

logger = logging.getLogger(__name__)


class CitationVerificationError(Exception):
    """Exception raised for failures during citation verification operations."""


class CitationVerificationService:
    """Orchestrates claim extraction, multi-signal evidence verification, and groundedness aggregation."""

    def __init__(
        self,
        verifier: Optional[ClaimVerifier] = None,
        min_overlap_threshold: float = CITATION_MIN_OVERLAP_THRESHOLD,
        entity_match_threshold: float = CITATION_ENTITY_MATCH_THRESHOLD,
    ) -> None:
        self.verifier = verifier or ClaimVerifier(
            min_overlap_threshold=min_overlap_threshold,
            entity_match_threshold=entity_match_threshold,
        )

    def verify(
        self, query: str, answer: str, sources: Optional[List[Any]] = None
    ) -> CitationVerificationResult:
        """Execute complete post-generation citation verification on answer against sources.

        Parameters
        ----------
        query : str
            Input user question.
        answer : str
            Generated natural language answer text.
        sources : List[RAGSource], optional
            Retrieved candidate sources (with chunk text). Defaults to empty list.

        Returns
        -------
        CitationVerificationResult
            Comprehensive verification report.
        """
        if not query or not query.strip():
            raise CitationVerificationError("Query string must not be empty.")
        if not answer or not answer.strip():
            raise CitationVerificationError("Answer string must not be empty.")

        clean_query = query.strip()
        clean_answer = answer.strip()
        source_list = sources or []

        # -------------------------------------------------------------------
        # 1. Refusal Sentinel Evaluation
        # -------------------------------------------------------------------
        is_refusal = (
            INSUFFICIENT_INFO_SENTINEL.lower() in clean_answer.lower()
            or "sufficient information" in clean_answer.lower()
        )

        if is_refusal:
            # Clean refusal: 0 factual claims, 100% groundedness, 0 spurious citations
            spurious_count = sum(
                len(c.citation_indices) + len(c.cited_chunk_ids)
                for c in extract_claims(clean_answer)
            )
            return CitationVerificationResult(
                query=clean_query,
                answer=clean_answer,
                answer_type=AnswerType.INSUFFICIENT_EVIDENCE,
                refusal_correct=True,
                claims=[],
                total_claims=0,
                grounded_claims_count=0,
                ungrounded_claims_count=0,
                claim_groundedness_rate=1.0,
                is_fully_grounded=True,
                total_citations=spurious_count,
                correct_citations_count=0,
                citation_precision=None if spurious_count == 0 else 0.0,
                citation_recall=1.0,
                citation_f1=None if spurious_count == 0 else 0.0,
                is_fully_cited=True,
                spurious_citations_count=spurious_count,
                invalid_citations_count=0,
                hallucinated_claims_count=0,
                unsupported_claims_count=0,
                wrong_citations_count=0,
                missing_citations_count=0,
            )

        # -------------------------------------------------------------------
        # 2. Extract Claims
        # -------------------------------------------------------------------
        claims = extract_claims(clean_answer)

        if not claims:
            return CitationVerificationResult(
                query=clean_query,
                answer=clean_answer,
                answer_type=AnswerType.GROUNDED_ANSWER,
                refusal_correct=None,
                claims=[],
                total_claims=0,
                grounded_claims_count=0,
                ungrounded_claims_count=0,
                claim_groundedness_rate=1.0,
                is_fully_grounded=True,
                total_citations=0,
                correct_citations_count=0,
                citation_precision=None,
                citation_recall=1.0,
                citation_f1=None,
                is_fully_cited=True,
            )

        # -------------------------------------------------------------------
        # 3. Verify Each Claim
        # -------------------------------------------------------------------
        claim_verifications: List[ClaimVerification] = [
            self.verifier.verify_claim(claim, source_list) for claim in claims
        ]

        # -------------------------------------------------------------------
        # 4. Aggregate Metrics
        # -------------------------------------------------------------------
        factual_verifications = [cv for cv in claim_verifications if cv.claim.is_factual]
        total_claims = len(factual_verifications)

        grounded_claims_count = sum(1 for cv in factual_verifications if cv.is_grounded)
        ungrounded_claims_count = total_claims - grounded_claims_count
        claim_groundedness_rate = (
            (grounded_claims_count / total_claims) if total_claims > 0 else 1.0
        )
        is_fully_grounded = ungrounded_claims_count == 0

        # Citation metrics
        cited_verifications = [cv for cv in factual_verifications if cv.has_citation]
        total_citations = sum(
            len(cv.claim.citation_indices) + len(cv.claim.cited_chunk_ids)
            for cv in factual_verifications
        )
        correct_citations_count = sum(
            1 for cv in factual_verifications if cv.has_citation and cv.is_citation_correct
        )

        grounded_verifications = [cv for cv in factual_verifications if cv.is_grounded]
        grounded_and_correctly_cited = [
            cv for cv in grounded_verifications if cv.has_citation and cv.is_citation_correct
        ]

        citation_recall = (
            (len(grounded_and_correctly_cited) / len(grounded_verifications))
            if grounded_verifications
            else 1.0
        )

        if total_citations == 0:
            citation_precision = None
            citation_f1 = None
        else:
            citation_precision = round(correct_citations_count / total_citations, 4)
            if (citation_precision + citation_recall) > 0.0:
                citation_f1 = round(
                    (2.0 * citation_precision * citation_recall)
                    / (citation_precision + citation_recall),
                    4,
                )
            else:
                citation_f1 = 0.0

        # Failure mode breakdown
        invalid_citations_count = sum(
            1
            for cv in factual_verifications
            if cv.unsupported_reason == UnsupportedReason.INVALID_CITATION_INDEX
        )
        hallucinated_claims_count = sum(
            1
            for cv in factual_verifications
            if cv.unsupported_reason
            in {
                UnsupportedReason.TEMPORAL_MISMATCH,
                UnsupportedReason.NUMERICAL_MISMATCH,
                UnsupportedReason.ENTITY_MISMATCH,
                UnsupportedReason.CONTRADICTION,
            }
        )
        unsupported_claims_count = sum(
            1
            for cv in factual_verifications
            if cv.unsupported_reason == UnsupportedReason.NO_SUPPORTING_SOURCE
        )
        wrong_citations_count = sum(
            1
            for cv in factual_verifications
            if cv.unsupported_reason == UnsupportedReason.WRONG_CITATION
        )
        missing_citations_count = sum(
            1
            for cv in factual_verifications
            if cv.unsupported_reason == UnsupportedReason.MISSING_CITATION
        )

        is_fully_cited = (
            (len(grounded_and_correctly_cited) == len(grounded_verifications))
            and (wrong_citations_count == 0)
            and (invalid_citations_count == 0)
            and (missing_citations_count == 0)
        ) if grounded_verifications else (wrong_citations_count == 0 and invalid_citations_count == 0)

        # Spurious citations on non-factual claims (e.g. greetings or pleasantries with citations)
        spurious_citations_count = sum(
            len(cv.claim.citation_indices) + len(cv.claim.cited_chunk_ids)
            for cv in claim_verifications
            if not cv.claim.is_factual
        )

        # Categorize AnswerType
        if is_fully_grounded:
            answer_type = AnswerType.GROUNDED_ANSWER
        elif hallucinated_claims_count > 0:
            answer_type = AnswerType.FABRICATED_ANSWER
        else:
            answer_type = AnswerType.UNSUPPORTED_ANSWER

        return CitationVerificationResult(
            query=clean_query,
            answer=clean_answer,
            answer_type=answer_type,
            refusal_correct=None,
            claims=claim_verifications,
            total_claims=total_claims,
            grounded_claims_count=grounded_claims_count,
            ungrounded_claims_count=ungrounded_claims_count,
            claim_groundedness_rate=round(claim_groundedness_rate, 4),
            is_fully_grounded=is_fully_grounded,
            total_citations=total_citations,
            correct_citations_count=correct_citations_count,
            citation_precision=(
                round(citation_precision, 4) if citation_precision is not None else None
            ),
            citation_recall=round(citation_recall, 4),
            citation_f1=round(citation_f1, 4) if citation_f1 is not None else None,
            is_fully_cited=is_fully_cited,

            spurious_citations_count=spurious_citations_count,
            invalid_citations_count=invalid_citations_count,
            hallucinated_claims_count=hallucinated_claims_count,
            unsupported_claims_count=unsupported_claims_count,
            wrong_citations_count=wrong_citations_count,
            missing_citations_count=missing_citations_count,
        )
