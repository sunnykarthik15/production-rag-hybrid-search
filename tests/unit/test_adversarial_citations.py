"""Unit tests for Phase 7.2 Adversarial Citation Verification and Synthetic Perturbations."""

from __future__ import annotations

import pytest

from app.citations.models import AnswerType, UnsupportedReason
from app.citations.service import CitationVerificationService
from app.rag.models import RAGSource
from scripts.evaluate_adversarial_citations import build_adversarial_dataset, run_adversarial_benchmark


def _make_source(chunk_id: str, text: str, rank: int = 1) -> RAGSource:
    return RAGSource(
        chunk_id=chunk_id,
        document_id="DOC001",
        title="Nexora Documentation",
        department="Operations",
        rank=rank,
        score=2.5,
        text=text,
    )


class TestAdversarialCitationVerification:
    """Test suite covering the 20 adversarial failure modes and perturbation categories."""

    @pytest.fixture
    def service(self) -> CitationVerificationService:
        return CitationVerificationService(min_overlap_threshold=0.40)

    def test_all_20_adversarial_benchmark_cases_pass(self) -> None:
        """Verify that all 20 categories (A through T) in the adversarial suite pass expectations."""
        benchmark_results = run_adversarial_benchmark()
        assert benchmark_results["metrics"]["accuracy"] == 1.0
        assert benchmark_results["metrics"]["precision"] == 1.0
        assert benchmark_results["metrics"]["recall"] == 1.0
        assert benchmark_results["metrics"]["f1"] == 1.0
        assert benchmark_results["confusion_matrix"]["FP"] == 0
        assert benchmark_results["confusion_matrix"]["FN"] == 0

    @pytest.mark.parametrize("case", build_adversarial_dataset(), ids=lambda c: c.case_id)
    def test_individual_adversarial_case(
        self, service: CitationVerificationService, case: Any
    ) -> None:
        """Test each individual adversarial case directly against the verification service."""
        result = service.verify(case.query, case.answer, sources=case.sources)

        pred_supported = (
            result.is_fully_grounded
            and result.is_fully_cited
            and result.ungrounded_claims_count == 0
            and result.missing_citations_count == 0
            and result.wrong_citations_count == 0
            and result.invalid_citations_count == 0
        )

        assert pred_supported == case.expected_supported, (
            f"Case {case.case_id} ({case.category}) failed: expected supported={case.expected_supported}, "
            f"got pred_supported={pred_supported}. Result: {result.model_dump()}"
        )

        if not case.expected_supported and result.claims:
            first_claim = result.claims[0]
            if case.expected_reason != UnsupportedReason.NONE:
                assert first_claim.unsupported_reason == case.expected_reason or result.claims[-1].unsupported_reason == case.expected_reason

    def test_benign_paraphrase_vs_temporal_mutation(self, service: CitationVerificationService) -> None:
        """Contrast test: Benign paraphrase is SUPPORTED, while weekday mutation is REJECTED."""
        source = _make_source(
            "DOC001_CHUNK_01",
            "Applications must be submitted before Friday at 5 PM.",
        )

        # 1. Benign paraphrase
        paraphrase_ans = "The application should be submitted prior to Friday at 5 PM [Source 1]."
        res_paraphrase = service.verify("When are applications due?", paraphrase_ans, sources=[source])
        assert res_paraphrase.is_fully_grounded is True
        assert res_paraphrase.is_fully_cited is True
        assert res_paraphrase.citation_precision == 1.0

        # 2. Weekday mutation (Wednesday vs Friday)
        mutation_ans = "The application should be submitted prior to Wednesday at 5 PM [Source 1]."
        res_mutation = service.verify("When are applications due?", mutation_ans, sources=[source])
        assert res_mutation.is_fully_grounded is False
        assert res_mutation.claims[0].unsupported_reason == UnsupportedReason.TEMPORAL_MISMATCH

    def test_benign_paraphrase_vs_numeric_mutation(self, service: CitationVerificationService) -> None:
        """Contrast test: Benign paraphrase is SUPPORTED, while fee mutation is REJECTED."""
        source = _make_source(
            "DOC001_CHUNK_01",
            "The registration fee is $50.00 per participant.",
        )

        # 1. Benign paraphrase
        paraphrase_ans = "Each participant is charged a registration fee of $50.00 [Source 1]."
        res_paraphrase = service.verify("What is the fee?", paraphrase_ans, sources=[source])
        assert res_paraphrase.is_fully_grounded is True

        # 2. Numeric mutation ($80 vs $50)
        mutation_ans = "Each participant is charged a registration fee of $80.00 [Source 1]."
        res_mutation = service.verify("What is the fee?", mutation_ans, sources=[source])
        assert res_mutation.is_fully_grounded is False
        assert res_mutation.claims[0].unsupported_reason == UnsupportedReason.NUMERICAL_MISMATCH

    def test_benign_paraphrase_vs_polarity_inversion(self, service: CitationVerificationService) -> None:
        """Contrast test: Benign paraphrase is SUPPORTED, while restriction inversion is REJECTED."""
        source = _make_source(
            "DOC001_CHUNK_01",
            "Access to the facility is restricted to authorized personnel.",
        )

        # 1. Benign paraphrase
        paraphrase_ans = "Entry to the facility is limited to authorized personnel [Source 1]."
        res_paraphrase = service.verify("Who can access?", paraphrase_ans, sources=[source])
        assert res_paraphrase.is_fully_grounded is True

        # 2. Polarity inversion (unrestricted vs restricted)
        mutation_ans = "Entry to the facility is unrestricted for all personnel [Source 1]."
        res_mutation = service.verify("Who can access?", mutation_ans, sources=[source])
        assert res_mutation.is_fully_grounded is False
        assert res_mutation.claims[0].unsupported_reason == UnsupportedReason.CONTRADICTION

    def test_benign_paraphrase_vs_negation_insertion(self, service: CitationVerificationService) -> None:
        """Contrast test: Negation insertion is detected as contradiction."""
        source = _make_source(
            "DOC001_CHUNK_01",
            "Building access requires supervisor approval and badge validation.",
        )

        # 1. Benign paraphrase
        paraphrase_ans = "Building entry requires supervisor approval and badge validation [Source 1]."
        res_paraphrase = service.verify("What is required?", paraphrase_ans, sources=[source])
        assert res_paraphrase.is_fully_grounded is True

        # 2. Negation insertion
        mutation_ans = "Building access does not require supervisor approval [Source 1]."
        res_mutation = service.verify("What is required?", mutation_ans, sources=[source])
        assert res_mutation.is_fully_grounded is False
        assert res_mutation.claims[0].unsupported_reason == UnsupportedReason.CONTRADICTION

    def test_benign_paraphrase_vs_negation_removal(self, service: CitationVerificationService) -> None:
        """Contrast test: Negation removal is detected as contradiction."""
        source = _make_source(
            "DOC001_CHUNK_01",
            "Visitor parking is prohibited in Lot B without an official permit.",
        )

        mutation_ans = "Visitor parking is permitted in Lot B without an official permit [Source 1]."
        res_mutation = service.verify("Is parking allowed?", mutation_ans, sources=[source])
        assert res_mutation.is_fully_grounded is False
        assert res_mutation.claims[0].unsupported_reason == UnsupportedReason.CONTRADICTION

    def test_distractor_citation_rejected(self, service: CitationVerificationService) -> None:
        """Distractor chunk cited alongside valid source fails citation correctness."""
        s1 = _make_source("DOC001_CHUNK_01", "Routine server maintenance occurs every Sunday at 2 AM.", rank=1)
        s2 = _make_source("DOC002_CHUNK_01", "Cafeteria menus rotate on a weekly basis every Monday.", rank=2)

        ans = "Routine server maintenance occurs every Sunday at 2 AM [Source 1, 2]."
        result = service.verify("When is server maintenance?", ans, sources=[s1, s2])
        assert result.is_fully_cited is False
        assert result.wrong_citations_count >= 1

    def test_zero_denominator_confusion_matrix_safety(self) -> None:
        """Zero denominator handling in confusion matrix must return None/N/A, never fake 1.0."""
        from scripts.evaluate_adversarial_citations import run_adversarial_benchmark

        # Benchmark run returns valid metrics container
        res = run_adversarial_benchmark(threshold=0.40)
        assert res["precision"] is not None
        assert res["recall"] is not None
        assert res["f1"] is not None
        assert res["confusion_matrix"]["TP"] > 0
        assert res["confusion_matrix"]["TN"] > 0

    def test_threshold_calibration_grid_execution(self) -> None:
        """Threshold calibration across 0.20-0.70 executes and recommends 0.40."""
        from scripts.calibrate_thresholds import run_calibration

        calib = run_calibration()
        assert calib["recommended_threshold"] == 0.40
        assert len(calib["calibration_results"]) == 11
        assert calib["best_f1"] == 1.0
