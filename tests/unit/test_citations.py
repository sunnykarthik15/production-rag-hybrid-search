"""Unit test suite for Phase 7 Citation Verification, Answer Groundedness, and Attribution."""

from __future__ import annotations

import pytest

from app.citations.extractor import (
    extract_claims,
    extract_codes,
    extract_dates,
    extract_entities,
    extract_numbers,
    extract_polarity_cues,
    extract_times,
    extract_weekdays,
    parse_citations_from_text,
    split_sentences,
)
from app.citations.models import (
    AnswerType,
    CitationVerificationResult,
    Claim,
    ClaimVerification,
    UnsupportedReason,
)
from app.citations.service import (
    CitationVerificationError,
    CitationVerificationService,
)
from app.citations.verifier import ClaimVerifier
from app.rag.models import RAGSource


def _make_source(
    chunk_id: str, text: str, rank: int = 1, score: float = 2.5
) -> RAGSource:
    """Helper factory for RAGSource with preserved chunk text."""
    return RAGSource(
        chunk_id=chunk_id,
        document_id="DOC001",
        title="Nexora Campus Operations Manual",
        department="Operations",
        rank=rank,
        score=score,
        text=text,
    )


# ===========================================================================
# 1. Extractor and Parsing Tests
# ===========================================================================


class TestExtractorAndParsers:
    """Test suite for sentence segmentation, citation parsing, and constraint extraction."""

    def test_split_sentences_basic(self) -> None:
        """Test sentence splitting on standard sentences."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_split_sentences_protected_abbreviations_and_codes(self) -> None:
        """Abbreviations (e.g., Dr., i.e.) and codes (NX-FAC-100) must not cause premature splits."""
        text = (
            "The supervisor Dr. Smith approved code NX-FAC-100 for campus use. "
            "Fee is $50.00 per student (e.g. for library access)."
        )
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "NX-FAC-100" in sentences[0]
        assert "$50.00" in sentences[1]
        assert "e.g." in sentences[1]

    def test_parse_citations_source_format(self) -> None:
        """Test parsing [Source 1] and [Source 1, 2] markers."""
        clean, indices, chunk_ids = parse_citations_from_text(
            "Library is open from 9 AM to 5 PM [Source 1, 2]."
        )
        assert clean == "Library is open from 9 AM to 5 PM"
        assert indices == [1, 2]
        assert chunk_ids == []

    def test_parse_citations_bracket_number_format(self) -> None:
        """Test parsing numeric citations like [1] and [2]."""
        clean, indices, chunk_ids = parse_citations_from_text(
            "Access requires registration [1] and badge verification [3]."
        )
        assert "Access requires registration" in clean
        assert indices == [1, 3]

    def test_parse_citations_chunk_id_format(self) -> None:
        """Test parsing explicit chunk ID citations like [DOC001_CHUNK_01]."""
        clean, indices, chunk_ids = parse_citations_from_text(
            "Facilities operate under module NX-FAC-100 [DOC001_CHUNK_01]."
        )
        assert "Facilities operate under module NX-FAC-100" in clean
        assert chunk_ids == ["DOC001_CHUNK_01"]

    def test_extract_entities_comprehensive(self) -> None:
        """Verify extraction of codes, dates, weekdays, times, and numbers."""
        text = "Under NX-FAC-100, on Friday Dec 15 at 9 AM, the fee is $50.00."
        entities = extract_entities(text)

        assert "NX-FAC-100" in entities["codes"]
        assert any("dec 15" in d for d in entities["dates"])
        assert "friday" in entities["weekdays"]
        assert any("9am" in t.replace(" ", "") for t in entities["times"])
        assert any("50" in n for n in entities["numbers"])

    def test_extract_polarity_cues(self) -> None:
        """Verify extraction of negation and restriction cues."""
        text1 = "Access is restricted and visitors are not allowed."
        text2 = "Entry is unrestricted and visitors are permitted."

        cues1 = extract_polarity_cues(text1)
        cues2 = extract_polarity_cues(text2)

        assert cues1["has_negation"] is True
        assert "restricted" in cues1["negation_terms"] or "not" in cues1["negation_terms"]

        assert "unrestricted" in cues2["affirmative_terms"] or "permitted" in cues2["affirmative_terms"]

    def test_split_sentences_punctuation_adjacent_citations(self) -> None:
        """Both 'The deadline is Friday. [Source 1]' and 'The deadline is Friday [Source 1].' remain attached."""
        text1 = "The application deadline is Friday. [Source 1] The fee is $50. [Source 2]"
        sentences1 = split_sentences(text1)
        assert len(sentences1) == 2
        assert "[Source 1]" in sentences1[0]
        assert "[Source 2]" in sentences1[1]

        text2 = "The application deadline is Friday [Source 1]. The fee is $50 [Source 2]."
        sentences2 = split_sentences(text2)
        assert len(sentences2) == 2
        assert "[Source 1]" in sentences2[0]
        assert "[Source 2]" in sentences2[1]

    def test_parse_citations_ordering(self) -> None:
        """Citation tags with multi-source ordering [Source 2, 1] or [Source 1, Source 2] parse properly."""
        clean1, idx1, _ = parse_citations_from_text("Access requires badge [Source 2, 1].")
        assert clean1 == "Access requires badge"
        assert idx1 == [1, 2]

        clean2, idx2, _ = parse_citations_from_text("Access requires badge [Source 1, Source 2].")
        assert clean2 == "Access requires badge"
        assert idx2 == [1, 2]


# ===========================================================================
# 2. Multi-Signal Claim Verifier Tests (Required Failure Modes)
# ===========================================================================


class TestClaimVerifierFailureModes:
    """Test suite for ClaimVerifier detecting specific factual discrepancies."""

    @pytest.fixture
    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier(min_overlap_threshold=0.40, entity_match_threshold=1.0)

    def test_grounded_claim_positive_match(self, verifier: ClaimVerifier) -> None:
        """Positive test: Fully grounded claim supported by matching source chunk."""
        source_text = "The library operating hours are 9 AM to 5 PM Monday through Friday under NX-LIB-101."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="The library operating hours are 9 AM to 5 PM Monday through Friday under NX-LIB-101.",
            raw_text="The library operating hours are 9 AM to 5 PM Monday through Friday under NX-LIB-101 [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is True
        assert verification.is_citation_correct is True
        assert verification.support_score >= 0.8
        assert verification.unsupported_reason == UnsupportedReason.NONE
        assert "DOC001_CHUNK_01" in verification.supporting_chunk_ids

    def test_failure_mode_fabricated_weekday(self, verifier: ClaimVerifier) -> None:
        """Failure mode: High lexical overlap but wrong weekday (Wednesday vs Friday)."""
        source_text = "Applications must be submitted before Friday at 5 PM."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        # Answer substitutes Friday with Wednesday
        claim = Claim(
            claim_id="C001",
            text="Applications must be submitted before Wednesday at 5 PM.",
            raw_text="Applications must be submitted before Wednesday at 5 PM [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.is_citation_correct is False
        assert verification.unsupported_reason == UnsupportedReason.TEMPORAL_MISMATCH
        assert verification.signals["entity_match"] == 0.0

    def test_failure_mode_fabricated_date(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Fabricated calendar date (Dec 15 vs Dec 12)."""
        source_text = "System maintenance is scheduled for Dec 12 on all central servers."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="System maintenance is scheduled for Dec 15 on all central servers.",
            raw_text="System maintenance is scheduled for Dec 15 on all central servers [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason == UnsupportedReason.TEMPORAL_MISMATCH

    def test_failure_mode_fabricated_number(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Fabricated fee / numeric value ($80 vs $50)."""
        source_text = "The annual parking registration fee is $50 per vehicle."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="The annual parking registration fee is $80 per vehicle.",
            raw_text="The annual parking registration fee is $80 per vehicle [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason == UnsupportedReason.NUMERICAL_MISMATCH

    def test_failure_mode_polarity_inversion_contradiction(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Contradictory assertion (unrestricted access vs restricted access)."""
        source_text = "Access to the server room is strictly restricted to authorized staff."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="Access to the server room is unrestricted for all staff.",
            raw_text="Access to the server room is unrestricted for all staff [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason == UnsupportedReason.CONTRADICTION

    def test_failure_mode_unsupported_claim(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Factual assertion not mentioned anywhere in retrieved chunks."""
        source_text = "The cafeteria serves breakfast from 7 AM to 10 AM daily."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="Quantum propulsion drones deliver packages across the quad.",
            raw_text="Quantum propulsion drones deliver packages across the quad [Source 1].",
            citation_indices=[1],
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason in {
            UnsupportedReason.NO_SUPPORTING_SOURCE,
            UnsupportedReason.ENTITY_MISMATCH,
        }

    def test_failure_mode_missing_citation(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Grounded claim that omits citation markers."""
        source_text = "The library provides 24/7 online catalog search access."
        source = _make_source("DOC001_CHUNK_01", source_text, rank=1)

        claim = Claim(
            claim_id="C001",
            text="The library provides 24/7 online catalog search access.",
            raw_text="The library provides 24/7 online catalog search access.",
            citation_indices=[],  # Missing citation
        )

        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is True
        assert verification.has_citation is False
        assert verification.is_citation_correct is False
        assert verification.unsupported_reason == UnsupportedReason.MISSING_CITATION

    def test_failure_mode_wrong_citation(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Claim is supported by Source 1, but cites Source 2."""
        source1 = _make_source("DOC001_CHUNK_01", "Library hours are 9 AM to 5 PM.", rank=1)
        source2 = _make_source("DOC002_CHUNK_01", "Parking permits cost $50 annually.", rank=2)

        # Claim asserts library hours but cites [Source 2]
        claim = Claim(
            claim_id="C001",
            text="Library hours are 9 AM to 5 PM.",
            raw_text="Library hours are 9 AM to 5 PM [Source 2].",
            citation_indices=[2],
        )

        verification = verifier.verify_claim(claim, [source1, source2])

        assert verification.is_grounded is True  # Grounded in context
        assert verification.is_citation_correct is False  # But citation to Source 2 was wrong
        assert verification.unsupported_reason == UnsupportedReason.WRONG_CITATION
        assert "DOC001_CHUNK_01" in verification.supporting_chunk_ids

    def test_failure_mode_invalid_citation_index(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Citation cites [Source 99] when only 2 sources exist."""
        source1 = _make_source("DOC001_CHUNK_01", "Library hours are 9 AM to 5 PM.", rank=1)

        claim = Claim(
            claim_id="C001",
            text="Library hours are 9 AM to 5 PM.",
            raw_text="Library hours are 9 AM to 5 PM [Source 99].",
            citation_indices=[99],
        )

        verification = verifier.verify_claim(claim, [source1])

        assert verification.is_grounded is False
        assert verification.is_citation_correct is False
        assert verification.unsupported_reason == UnsupportedReason.INVALID_CITATION_INDEX

    def test_failure_mode_fabricated_time(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Fabricated time constraint (10 PM vs 5 PM)."""
        source = _make_source("DOC001_CHUNK_01", "The computer lab closes at 5 PM daily.", rank=1)
        claim = Claim(
            claim_id="C001",
            text="The computer lab closes at 10 PM daily.",
            raw_text="The computer lab closes at 10 PM daily [Source 1].",
            citation_indices=[1],
        )
        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason == UnsupportedReason.TEMPORAL_MISMATCH

    def test_failure_mode_out_of_range_indices_zero(self, verifier: ClaimVerifier) -> None:
        """Failure mode: Index 0 or negative index is out-of-range."""
        source = _make_source("DOC001_CHUNK_01", "The computer lab closes at 5 PM daily.", rank=1)
        claim = Claim(
            claim_id="C001",
            text="The computer lab closes at 5 PM daily.",
            raw_text="The computer lab closes at 5 PM daily [Source 0].",
            citation_indices=[0],
        )
        verification = verifier.verify_claim(claim, [source])

        assert verification.is_grounded is False
        assert verification.unsupported_reason == UnsupportedReason.INVALID_CITATION_INDEX


# ===========================================================================
# 3. CitationVerificationService End-to-End Tests
# ===========================================================================


class TestCitationVerificationService:
    """Test suite for CitationVerificationService high-level orchestration."""

    @pytest.fixture
    def service(self) -> CitationVerificationService:
        return CitationVerificationService()

    def test_correct_refusal_unanswerable_query(self, service: CitationVerificationService) -> None:
        """Refusal sentinel on unanswerable query returns groundedness=1.0, 0 claims, refusal_correct=True."""
        refusal_text = "I do not have sufficient information in the provided context to answer this question."
        result = service.verify("What is the warp drive speed?", refusal_text, sources=[])

        assert isinstance(result, CitationVerificationResult)
        assert result.answer_type == AnswerType.INSUFFICIENT_EVIDENCE
        assert result.refusal_correct is True
        assert result.claim_groundedness_rate == 1.0
        assert result.total_claims == 0
        assert result.is_fully_grounded is True
        assert result.spurious_citations_count == 0
        assert result.citation_precision is None
        assert result.citation_f1 is None

    def test_fabricated_answer_to_unanswerable_query(
        self, service: CitationVerificationService
    ) -> None:
        """Fabricated answer to unanswerable query with irrelevant sources is flagged as fabricated."""
        irrelevant_source = _make_source(
            "DOC010_CHUNK_01", "Nexora standard waste disposal schedule."
        )
        fabricated_answer = (
            "Warp drive speeds exceed warp 9.5 across all campus shuttles [Source 1]."
        )

        result = service.verify(
            "What is the warp drive speed?", fabricated_answer, sources=[irrelevant_source]
        )

        assert result.answer_type == AnswerType.FABRICATED_ANSWER
        assert result.refusal_correct is None
        assert result.claim_groundedness_rate == 0.0
        assert result.is_fully_grounded is False
        assert result.hallucinated_claims_count >= 1

    def test_multi_claim_multi_source_answer(self, service: CitationVerificationService) -> None:
        """Multi-sentence answer where Claim 1 cites Source 1 and Claim 2 cites Source 2."""
        s1 = _make_source("DOC001_CHUNK_01", "Library hours are 9 AM to 5 PM daily.", rank=1)
        s2 = _make_source("DOC002_CHUNK_01", "Parking permits cost $50 per semester.", rank=2)

        answer = (
            "Library hours are 9 AM to 5 PM daily [Source 1]. "
            "Parking permits cost $50 per semester [Source 2]."
        )

        result = service.verify("What are the hours and permit costs?", answer, sources=[s1, s2])

        assert result.total_claims == 2
        assert result.grounded_claims_count == 2
        assert result.claim_groundedness_rate == 1.0
        assert result.citation_precision == 1.0
        assert result.citation_recall == 1.0
        assert result.is_fully_grounded is True
        assert result.is_fully_cited is True

    def test_case_a_correct_citation(self, service: CitationVerificationService) -> None:
        """Case A: Correct citation - answer claims deadline Friday citing Source 1 which states Friday."""
        s1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
        answer = "The application deadline is Friday. [Source 1]"

        result = service.verify("When is the application deadline?", answer, sources=[s1])

        assert result.is_fully_grounded is True
        assert result.total_citations == 1
        assert result.correct_citations_count == 1
        assert result.citation_precision == 1.0
        assert result.citation_recall == 1.0
        assert result.citation_f1 == 1.0
        assert result.missing_citations_count == 0
        assert result.wrong_citations_count == 0

    def test_case_b_wrong_citation(self, service: CitationVerificationService) -> None:
        """Case B: Wrong citation - answer claims deadline Friday citing Source 2 (which is parking permits)."""
        s1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
        s2 = _make_source("DOC002_CHUNK_01", "Parking permits cost $50 per semester.", rank=2)
        answer = "The application deadline is Friday. [Source 2]"

        result = service.verify("When is the application deadline?", answer, sources=[s1, s2])

        assert result.is_fully_grounded is True  # Grounded in s1
        assert result.total_citations == 1
        assert result.correct_citations_count == 0
        assert result.wrong_citations_count == 1
        assert result.citation_precision == 0.0
        assert result.citation_recall == 0.0

    def test_case_c_missing_citation(self, service: CitationVerificationService) -> None:
        """Case C: Missing citation - answer contains grounded claim with 0 citations (precision=None)."""
        s1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
        answer = "The application deadline is Friday."

        result = service.verify("When is the application deadline?", answer, sources=[s1])

        assert result.is_fully_grounded is True
        assert result.total_citations == 0
        assert result.correct_citations_count == 0
        assert result.missing_citations_count == 1
        assert result.citation_precision is None  # Precision is N/A / None when 0 citations made
        assert result.citation_recall == 0.0
        assert result.citation_f1 is None

    def test_case_d_invalid_citation(self, service: CitationVerificationService) -> None:
        """Case D: Invalid citation - answer cites Source 99 when only 1 source exists."""
        s1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
        answer = "The application deadline is Friday. [Source 99]"

        result = service.verify("When is the application deadline?", answer, sources=[s1])

        assert result.invalid_citations_count == 1
        assert result.total_citations == 1
        assert result.correct_citations_count == 0
        assert result.citation_precision == 0.0

    def test_case_e_multi_source_answer(self, service: CitationVerificationService) -> None:
        """Case E: Multi-source answer - Claim 1 cites Source 1 and Claim 2 cites Source 2."""
        s1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
        s2 = _make_source("DOC002_CHUNK_01", "The application fee is $50.", rank=2)
        answer = "The application deadline is Friday. [Source 1] The application fee is $50. [Source 2]"

        result = service.verify("What are the deadline and fee?", answer, sources=[s1, s2])

        assert result.total_claims == 2
        assert result.grounded_claims_count == 2
        assert result.total_citations == 2
        assert result.correct_citations_count == 2
        assert result.wrong_citations_count == 0
        assert result.missing_citations_count == 0
        assert result.citation_precision == 1.0
        assert result.citation_recall == 1.0
        assert result.citation_f1 == 1.0

    def test_empty_query_or_answer_raises_error(
        self, service: CitationVerificationService
    ) -> None:
        """Empty query or answer raises CitationVerificationError."""
        with pytest.raises(CitationVerificationError, match="empty"):
            service.verify("", "Some valid answer text.")

        with pytest.raises(CitationVerificationError, match="empty"):
            service.verify("Valid query", "")

    def test_pydantic_model_hardening(self) -> None:
        """Verify strict Pydantic model validations and constraints."""
        from pydantic import ValidationError

        # Claim text must not be empty
        with pytest.raises(ValidationError):
            Claim(claim_id="C001", text="", raw_text="raw")

        # RAGSource rank must be >= 1
        with pytest.raises(ValidationError):
            RAGSource(
                chunk_id="C1",
                document_id="D1",
                title="T1",
                department="Ops",
                rank=0,
                score=1.0,
            )

        # Support score ge=0.0, le=1.0
        claim = Claim(claim_id="C001", text="Test claim", raw_text="Test claim [1]")
        with pytest.raises(ValidationError):
            ClaimVerification(
                claim=claim,
                is_grounded=True,
                is_citation_correct=True,
                has_citation=True,
                support_score=1.5,
            )

