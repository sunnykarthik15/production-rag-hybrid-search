#!/usr/bin/env python3
"""
scripts/evaluate_adversarial_citations.py
==========================================
Deterministic Adversarial Evaluation Suite for Citation Verification & Groundedness.

Evaluates the citation verifier against 20 controlled perturbation categories (A through T)
using real Nexora source chunks:
  A. Faithful paraphrase
  B. Exact source statement
  C. Missing citation
  D. Wrong citation
  E. Invalid citation index
  F. Fabricated number
  G. Fabricated date
  H. Fabricated weekday
  I. Fabricated time
  J. Polarity inversion
  K. Negation insertion
  L. Negation removal
  M. Unsupported entity
  N. Unsupported claim
  O. Partial claim
  P. Multi-source correct attribution
  Q. Multi-source wrong attribution
  R. Distractor citation
  S. Contradictory numerical value
  T. Contradictory temporal value

Computes:
  - 2x2 Confusion Matrix (TP, TN, FP, FN)
  - Precision, Recall, F1, False Positive Rate (FPR), False Negative Rate (FNR)
  - Detailed per-category breakdown
  - Comprehensive failure reason distribution

Outputs machine-readable results to `results/adversarial_citation_eval.json`.

Usage:
    python scripts/evaluate_adversarial_citations.py
"""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.citations.models import UnsupportedReason
from app.citations.service import CitationVerificationService
from app.config import (
    CITATION_ENTITY_MATCH_THRESHOLD,
    CITATION_MIN_OVERLAP_THRESHOLD,
    RESULTS_DIR,
)
from app.rag.models import RAGSource

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_OUTPUT_PATH = RESULTS_DIR / "adversarial_citation_eval.json"
COMPAT_OUTPUT_PATH = RESULTS_DIR / "adversarial_citations_eval.json"
SEP = "=" * 75


def _make_source(
    chunk_id: str,
    text: str,
    rank: int = 1,
    title: str = "Nexora Operations Manual",
    dept: str = "Operations",
) -> RAGSource:
    return RAGSource(
        chunk_id=chunk_id,
        document_id="DOC001",
        title=title,
        department=dept,
        rank=rank,
        score=3.0 - (rank * 0.2),
        text=text,
    )


@dataclass
class AdversarialCase:
    case_id: str
    category: str
    description: str
    query: str
    answer: str
    sources: List[RAGSource]
    expected_supported: bool
    expected_reason: UnsupportedReason


def build_adversarial_dataset() -> List[AdversarialCase]:
    """Construct deterministic test cases covering all 20 categories (A through T)."""
    cases: List[AdversarialCase] = []

    # A. Faithful paraphrase
    s_a = _make_source(
        "DOC001_CHUNK_01",
        "The library operating hours are 9 AM to 5 PM Monday through Friday under NX-LIB-101.",
    )
    cases.append(
        AdversarialCase(
            case_id="ADV_A01",
            category="A. Faithful paraphrase",
            description="Benign linguistic paraphrase with preserved facts and correct citation",
            query="What are the library hours?",
            answer="Library service hours are from 9 AM to 5 PM Monday through Friday governed by specification NX-LIB-101 [Source 1].",
            sources=[s_a],
            expected_supported=True,
            expected_reason=UnsupportedReason.NONE,
        )
    )

    # B. Exact source statement
    s_b = _make_source(
        "DOC001_CHUNK_01",
        "Annual parking registration permits cost $50 per vehicle.",
    )
    cases.append(
        AdversarialCase(
            case_id="ADV_B01",
            category="B. Exact source statement",
            description="Verbatim factual assertion directly from the retrieved chunk",
            query="What is the parking registration fee?",
            answer="Annual parking registration permits cost $50 per vehicle [Source 1].",
            sources=[s_b],
            expected_supported=True,
            expected_reason=UnsupportedReason.NONE,
        )
    )

    # C. Missing citation
    s_c = _make_source("DOC001_CHUNK_01", "The library provides 24/7 online catalog search access.")
    cases.append(
        AdversarialCase(
            case_id="ADV_C01",
            category="C. Missing citation",
            description="Grounded claim with citation markers completely omitted",
            query="What online access does the library provide?",
            answer="The library provides 24/7 online catalog search access.",
            sources=[s_c],
            expected_supported=False,
            expected_reason=UnsupportedReason.MISSING_CITATION,
        )
    )

    # D. Wrong citation
    s_d1 = _make_source("DOC001_CHUNK_01", "The application deadline is Friday at 5 PM.", rank=1)
    s_d2 = _make_source("DOC002_CHUNK_01", "The annual parking registration fee is $50 per vehicle.", rank=2)
    cases.append(
        AdversarialCase(
            case_id="ADV_D01",
            category="D. Wrong citation",
            description="True claim cites Source 2 instead of supporting Source 1",
            query="When is the application deadline?",
            answer="The application deadline is Friday at 5 PM [Source 2].",
            sources=[s_d1, s_d2],
            expected_supported=False,
            expected_reason=UnsupportedReason.WRONG_CITATION,
        )
    )

    # E. Invalid citation index
    s_e = _make_source("DOC001_CHUNK_01", "The computer lab closes at 5 PM daily.")
    cases.append(
        AdversarialCase(
            case_id="ADV_E01",
            category="E. Invalid citation index",
            description="Citation marker cites out-of-range Source 99",
            query="When does the computer lab close?",
            answer="The computer lab closes at 5 PM daily [Source 99].",
            sources=[s_e],
            expected_supported=False,
            expected_reason=UnsupportedReason.INVALID_CITATION_INDEX,
        )
    )

    # F. Fabricated number
    s_f = _make_source("DOC001_CHUNK_01", "The annual parking registration fee is $50 per vehicle.")
    cases.append(
        AdversarialCase(
            case_id="ADV_F01",
            category="F. Fabricated number",
            description="Mutated numeric amount ($80 vs $50)",
            query="What is the parking registration fee?",
            answer="The annual parking registration fee is $80 per vehicle [Source 1].",
            sources=[s_f],
            expected_supported=False,
            expected_reason=UnsupportedReason.NUMERICAL_MISMATCH,
        )
    )

    # G. Fabricated date
    s_g = _make_source("DOC001_CHUNK_01", "System maintenance is scheduled for Dec 12 on all central servers.")
    cases.append(
        AdversarialCase(
            case_id="ADV_G01",
            category="G. Fabricated date",
            description="Mutated date (Dec 15 vs Dec 12)",
            query="When is system maintenance scheduled?",
            answer="System maintenance is scheduled for Dec 15 on all central servers [Source 1].",
            sources=[s_g],
            expected_supported=False,
            expected_reason=UnsupportedReason.TEMPORAL_MISMATCH,
        )
    )

    # H. Fabricated weekday
    s_h = _make_source("DOC001_CHUNK_01", "Applications must be submitted before Friday at 5 PM.")
    cases.append(
        AdversarialCase(
            case_id="ADV_H01",
            category="H. Fabricated weekday",
            description="Mutated day of week (Wednesday vs Friday)",
            query="When must applications be submitted?",
            answer="Applications must be submitted before Wednesday at 5 PM [Source 1].",
            sources=[s_h],
            expected_supported=False,
            expected_reason=UnsupportedReason.TEMPORAL_MISMATCH,
        )
    )

    # I. Fabricated time
    s_i = _make_source("DOC001_CHUNK_01", "The computer lab closes at 5 PM daily.")
    cases.append(
        AdversarialCase(
            case_id="ADV_I01",
            category="I. Fabricated time",
            description="Mutated clock time (10 PM vs 5 PM)",
            query="What time does the computer lab close?",
            answer="The computer lab closes at 10 PM daily [Source 1].",
            sources=[s_i],
            expected_supported=False,
            expected_reason=UnsupportedReason.TEMPORAL_MISMATCH,
        )
    )

    # J. Polarity inversion
    s_j = _make_source("DOC001_CHUNK_01", "Access to the server room is strictly restricted to authorized staff.")
    cases.append(
        AdversarialCase(
            case_id="ADV_J01",
            category="J. Polarity inversion",
            description="Inverted restriction (unrestricted vs restricted)",
            query="Who can access the server room?",
            answer="Access to the server room is unrestricted for all staff [Source 1].",
            sources=[s_j],
            expected_supported=False,
            expected_reason=UnsupportedReason.CONTRADICTION,
        )
    )

    # K. Negation insertion
    s_k = _make_source("DOC001_CHUNK_01", "Building access requires supervisor approval and security badge validation.")
    cases.append(
        AdversarialCase(
            case_id="ADV_K01",
            category="K. Negation insertion",
            description="Negation inserted into affirmative source requirement",
            query="What is required for building access?",
            answer="Building access does not require supervisor approval [Source 1].",
            sources=[s_k],
            expected_supported=False,
            expected_reason=UnsupportedReason.CONTRADICTION,
        )
    )

    # L. Negation removal
    s_l = _make_source("DOC001_CHUNK_01", "Visitor parking is prohibited in Lot B without an official visitor permit.")
    cases.append(
        AdversarialCase(
            case_id="ADV_L01",
            category="L. Negation removal",
            description="Negation removed making prohibited action appear permitted",
            query="Is visitor parking allowed in Lot B without a permit?",
            answer="Visitor parking is permitted in Lot B without an official visitor permit [Source 1].",
            sources=[s_l],
            expected_supported=False,
            expected_reason=UnsupportedReason.CONTRADICTION,
        )
    )

    # M. Unsupported entity
    s_m = _make_source("DOC001_CHUNK_01", "Facilities management operates under module NX-FAC-100.")
    cases.append(
        AdversarialCase(
            case_id="ADV_M01",
            category="M. Unsupported entity",
            description="Substituted module code (NX-SEC-999 vs NX-FAC-100)",
            query="Which module governs facilities management?",
            answer="Facilities management operates under module NX-SEC-999 [Source 1].",
            sources=[s_m],
            expected_supported=False,
            expected_reason=UnsupportedReason.ENTITY_MISMATCH,
        )
    )

    # N. Unsupported claim
    s_n = _make_source("DOC001_CHUNK_01", "The cafeteria serves breakfast from 7 AM to 10 AM daily.")
    cases.append(
        AdversarialCase(
            case_id="ADV_N01",
            category="N. Unsupported claim",
            description="Completely hallucinated assertion not in context",
            query="How are meals delivered?",
            answer="Quantum propulsion drones deliver breakfast across the quad [Source 1].",
            sources=[s_n],
            expected_supported=False,
            expected_reason=UnsupportedReason.NO_SUPPORTING_SOURCE,
        )
    )

    # O. Partial claim
    s_o = _make_source("DOC001_CHUNK_01", "The cafeteria serves breakfast from 7 AM to 10 AM daily.")
    cases.append(
        AdversarialCase(
            case_id="ADV_O01",
            category="O. Partial claim",
            description="One grounded claim and one hallucinated claim",
            query="What cafeteria services exist?",
            answer="The cafeteria serves breakfast from 7 AM to 10 AM daily [Source 1]. Sub-orbital shuttles serve lunch on the rooftop [Source 1].",
            sources=[s_o],
            expected_supported=False,
            expected_reason=UnsupportedReason.NO_SUPPORTING_SOURCE,
        )
    )

    # P. Multi-source correct attribution
    s_p1 = _make_source("DOC001_CHUNK_01", "Library hours are 9 AM to 5 PM daily.", rank=1)
    s_p2 = _make_source("DOC002_CHUNK_01", "Parking permits cost $50 per semester.", rank=2)
    cases.append(
        AdversarialCase(
            case_id="ADV_P01",
            category="P. Multi-source correct attribution",
            description="Multi-sentence answer where each sentence correctly cites its respective source",
            query="What are the hours and permit costs?",
            answer="Library hours are 9 AM to 5 PM daily [Source 1]. Parking permits cost $50 per semester [Source 2].",
            sources=[s_p1, s_p2],
            expected_supported=True,
            expected_reason=UnsupportedReason.NONE,
        )
    )

    # Q. Multi-source wrong attribution
    s_q1 = _make_source("DOC001_CHUNK_01", "Library hours are 9 AM to 5 PM daily.", rank=1)
    s_q2 = _make_source("DOC002_CHUNK_01", "Parking permits cost $50 per semester.", rank=2)
    cases.append(
        AdversarialCase(
            case_id="ADV_Q01",
            category="Q. Multi-source wrong attribution",
            description="Multi-sentence answer where source citations are inverted",
            query="What are the hours and permit costs?",
            answer="Library hours are 9 AM to 5 PM daily [Source 2]. Parking permits cost $50 per semester [Source 1].",
            sources=[s_q1, s_q2],
            expected_supported=False,
            expected_reason=UnsupportedReason.WRONG_CITATION,
        )
    )

    # R. Distractor citation
    s_r1 = _make_source("DOC001_CHUNK_01", "Routine server maintenance occurs every Sunday at 2 AM.", rank=1)
    s_r2 = _make_source("DOC002_CHUNK_01", "Cafeteria menus rotate on a weekly basis every Monday.", rank=2)
    cases.append(
        AdversarialCase(
            case_id="ADV_R01",
            category="R. Distractor citation",
            description="Claim cites [Source 1, 2] but Source 2 is an irrelevant distractor chunk",
            query="When does routine server maintenance occur?",
            answer="Routine server maintenance occurs every Sunday at 2 AM [Source 1, 2].",
            sources=[s_r1, s_r2],
            expected_supported=False,
            expected_reason=UnsupportedReason.WRONG_CITATION,
        )
    )

    # S. Contradictory numerical value
    s_s = _make_source(
        "DOC001_CHUNK_01",
        "The chemical solution ratio is 3.5 ml per liter, costing $50.00 per vial.",
    )
    cases.append(
        AdversarialCase(
            case_id="ADV_S01",
            category="S. Contradictory numerical value",
            description="Mutated ratio (7.5 vs 3.5) and price ($90.00 vs $50.00)",
            query="What is the solution ratio and cost?",
            answer="The chemical solution ratio is 7.5 ml per liter, costing $90.00 per vial [Source 1].",
            sources=[s_s],
            expected_supported=False,
            expected_reason=UnsupportedReason.NUMERICAL_MISMATCH,
        )
    )

    # T. Contradictory temporal value
    s_t = _make_source(
        "DOC001_CHUNK_01",
        "System backup starts at 08:00 and concludes by 11:00 every Tuesday.",
    )
    cases.append(
        AdversarialCase(
            case_id="ADV_T01",
            category="T. Contradictory temporal value",
            description="Mutated time (14:00/18:00 vs 08:00/11:00) and day (Thursday vs Tuesday)",
            query="When does system backup take place?",
            answer="System backup starts at 14:00 and concludes by 18:00 every Thursday [Source 1].",
            sources=[s_t],
            expected_supported=False,
            expected_reason=UnsupportedReason.TEMPORAL_MISMATCH,
        )
    )

    return cases


def run_adversarial_benchmark(
    threshold: float = CITATION_MIN_OVERLAP_THRESHOLD,
) -> Dict[str, Any]:
    """Execute adversarial evaluation and compute confusion matrix and performance metrics."""
    service = CitationVerificationService(min_overlap_threshold=threshold)
    dataset = build_adversarial_dataset()

    tp = 0  # Actual Supported, Pred Supported
    fp = 0  # Actual Unsupported, Pred Supported (Type I Error)
    fn = 0  # Actual Supported, Pred Unsupported (Type II Error)
    tn = 0  # Actual Unsupported, Pred Unsupported (Correct Rejection)

    total_claims = 0
    grounded_claims = 0
    ungrounded_claims = 0
    correct_citations = 0
    incorrect_citations = 0
    missing_citations = 0
    invalid_citations = 0
    hallucinated_claims = 0
    unsupported_claims = 0

    failure_reasons: Dict[str, int] = {}
    category_results: Dict[str, Dict[str, Any]] = {}
    case_details: List[Dict[str, Any]] = []

    for case in dataset:
        result = service.verify(case.query, case.answer, sources=case.sources)

        # Aggregate claim-level statistics
        total_claims += result.total_claims
        grounded_claims += result.grounded_claims_count
        ungrounded_claims += result.ungrounded_claims_count
        correct_citations += result.correct_citations_count
        invalid_citations += result.invalid_citations_count
        missing_citations += result.missing_citations_count
        hallucinated_claims += result.hallucinated_claims_count
        unsupported_claims += result.unsupported_claims_count
        incorrect_citations += result.wrong_citations_count + result.invalid_citations_count

        # A response is predicted supported if it is fully grounded, fully cited, and has no verification defects
        pred_supported = (
            result.is_fully_grounded
            and result.is_fully_cited
            and result.ungrounded_claims_count == 0
            and result.missing_citations_count == 0
            and result.wrong_citations_count == 0
            and result.invalid_citations_count == 0
        )

        actual_supported = case.expected_supported

        if actual_supported and pred_supported:
            tp += 1
            classification = "TP"
        elif not actual_supported and not pred_supported:
            tn += 1
            classification = "TN"
        elif not actual_supported and pred_supported:
            fp += 1
            classification = "FP"
        else:
            fn += 1
            classification = "FN"

        passed_expectation = (actual_supported == pred_supported)

        # Diagnostic reasons tracking
        for c in result.claims:
            if c.unsupported_reason != UnsupportedReason.NONE:
                r_str = c.unsupported_reason.value
                failure_reasons[r_str] = failure_reasons.get(r_str, 0) + 1

        cat_entry = category_results.setdefault(
            case.category,
            {"total": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        )
        cat_entry["total"] += 1
        if passed_expectation:
            cat_entry["correct"] += 1
        if classification == "TP":
            cat_entry["tp"] += 1
        elif classification == "TN":
            cat_entry["tn"] += 1
        elif classification == "FP":
            cat_entry["fp"] += 1
        elif classification == "FN":
            cat_entry["fn"] += 1

        primary_reason = (
            result.claims[0].unsupported_reason.value
            if result.claims
            else "none"
        )

        case_details.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "description": case.description,
                "actual_supported": actual_supported,
                "pred_supported": pred_supported,
                "classification": classification,
                "passed": passed_expectation,
                "claims_count": result.total_claims,
                "grounded_rate": result.claim_groundedness_rate,
                "citation_precision": result.citation_precision,
                "citation_recall": result.citation_recall,
                "primary_reason": primary_reason,
            }
        )

    total_samples = len(dataset)

    # Zero-denominator safe metric calculations (None when undefined, NEVER fake 1.0)
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else None
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round((2.0 * precision * recall) / (precision + recall), 4)
    else:
        f1 = None
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else (0.0 if tn > 0 else None)
    fnr = round(fn / (tp + fn), 4) if (tp + fn) > 0 else (0.0 if tp > 0 else None)
    accuracy = round((tp + tn) / total_samples, 4) if total_samples > 0 else None

    # Paraphrase acceptance & adversarial rejection rates
    benign_cases = [c for c in case_details if c["actual_supported"]]
    adversarial_cases = [c for c in case_details if not c["actual_supported"]]

    benign_accepted = sum(1 for c in benign_cases if c["pred_supported"])
    paraphrase_acceptance_rate = (
        round(benign_accepted / len(benign_cases), 4) if benign_cases else None
    )

    adversarial_rejected = sum(1 for c in adversarial_cases if not c["pred_supported"])
    adversarial_rejection_rate = (
        round(adversarial_rejected / len(adversarial_cases), 4) if adversarial_cases else None
    )

    return {
        "threshold": threshold,
        "total_cases": total_samples,
        "total_claims": total_claims,
        "grounded_claims": grounded_claims,
        "ungrounded_claims": ungrounded_claims,
        "correct_citations": correct_citations,
        "incorrect_citations": incorrect_citations,
        "missing_citations": missing_citations,
        "invalid_citations": invalid_citations,
        "hallucinated_claims": hallucinated_claims,
        "unsupported_claims": unsupported_claims,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "accuracy": accuracy,
        "paraphrase_acceptance_rate": paraphrase_acceptance_rate,
        "adversarial_rejection_rate": adversarial_rejection_rate,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
        },
        "results_by_perturbation_category": category_results,
        "category_breakdown": category_results,
        "failure_reason_distribution": failure_reasons,
        "cases": case_details,
    }


def main() -> int:
    print(f"\n{SEP}")
    print("PHASE 7.2 ADVERSARIAL CITATION VERIFICATION BENCHMARK")
    print(f"{SEP}\n")

    results = run_adversarial_benchmark(threshold=CITATION_MIN_OVERLAP_THRESHOLD)
    cm = results["confusion_matrix"]

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f} ({v*100:.2f}%)" if v is not None else "N/A"

    def _fmt_raw(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    print("Confusion Matrix:")
    print("                      Actual Supported    Actual Unsupported")
    print(f"  Pred Supported      TP: {cm['TP']:<15} FP: {cm['FP']}")
    print(f"  Pred Unsupported    FN: {cm['FN']:<15} TN: {cm['TN']}\n")

    print("Verification Performance Metrics:")
    print(f"  Total Test Cases:            {results['total_cases']}")
    print(f"  Accuracy:                    {_fmt(results['accuracy'])}")
    print(f"  Precision:                   {_fmt(results['precision'])}")
    print(f"  Recall:                      {_fmt(results['recall'])}")
    print(f"  F1 Score:                    {_fmt_raw(results['f1'])}")
    print(f"  False Positive Rate (FPR):   {_fmt(results['false_positive_rate'])}")
    print(f"  False Negative Rate (FNR):   {_fmt(results['false_negative_rate'])}")
    print(f"  Paraphrase Acceptance Rate:  {_fmt(results['paraphrase_acceptance_rate'])}")
    print(f"  Adversarial Rejection Rate:  {_fmt(results['adversarial_rejection_rate'])}\n")

    print("Claim and Citation Accounting:")
    print(f"  Total Claims Evaluated:      {results['total_claims']}")
    print(f"  Grounded Claims:             {results['grounded_claims']}")
    print(f"  Ungrounded Claims:           {results['ungrounded_claims']}")
    print(f"  Correct Citations:           {results['correct_citations']}")
    print(f"  Incorrect/Wrong Citations:   {results['incorrect_citations']}")
    print(f"  Missing Citations:           {results['missing_citations']}")
    print(f"  Invalid Citation Indices:    {results['invalid_citations']}")
    print(f"  Hallucinated Assertions:     {results['hallucinated_claims']}")
    print(f"  Unsupported Claims:          {results['unsupported_claims']}\n")

    print("Category Breakdown (A through T):")
    print(f"{'-'*75}")
    print(f"{'Category':<45} {'Count':<8} {'Status':<10}")
    print(f"{'-'*75}")
    for cat, data in results["results_by_perturbation_category"].items():
        status = "PASS" if data["correct"] == data["total"] else f"FAIL ({data['correct']}/{data['total']})"
        print(f"{cat:<45} {data['total']:<8} {status:<10}")
    print(f"{'-'*75}\n")

    if results["failure_reason_distribution"]:
        print("Failure Reason Distribution:")
        for reason, count in sorted(results["failure_reason_distribution"].items(), key=lambda x: -x[1]):
            print(f"  - {reason:<25}: {count}")
        print()

    PRIMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRIMARY_OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    with COMPAT_OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Saved adversarial evaluation results to:\n  {PRIMARY_OUTPUT_PATH.relative_to(_REPO_ROOT)}\n")
    print(f"{SEP}")
    print("ADVERSARIAL EVALUATION COMPLETE")
    print(f"{SEP}\n")

    return 0 if results["accuracy"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
