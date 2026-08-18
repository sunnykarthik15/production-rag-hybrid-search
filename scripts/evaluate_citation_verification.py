#!/usr/bin/env python3
"""
scripts/evaluate_citation_verification.py
=========================================
CLI script to evaluate Phase 7 Citation Verification, Answer Groundedness, and Attribution
performance against the 150 evaluation questions in the Nexora dataset.

Calculates:
  - Retrieval Metrics: Recall@1, Recall@3, Recall@5, Recall@10, MRR
  - Groundedness Metrics: Claim Groundedness Rate, Fully Grounded Answer Rate
  - Citation Metrics: Citation Precision, Citation Recall, Citation F1
  - Unanswerable Metrics: Correct Refusal Rate, Spurious Citations, Fabrications
  - Failure Mode Diagnostics: Hallucinated, Unsupported, Wrong, Missing, Invalid Citations
  - Score Distributions: Min, Max, Mean, Std, P25, P50, P75, P90

Saves machine-readable evaluation results to `results/citation_verification_eval.json`.

Usage
-----
    python scripts/evaluate_citation_verification.py

Exit codes
----------
    0 — Evaluation completed successfully.
    1 — Evaluation failed.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.citations.models import AnswerType
from app.citations.service import CitationVerificationService
from app.config import (
    CITATION_ENTITY_MATCH_THRESHOLD,
    CITATION_MIN_OVERLAP_THRESHOLD,
    LLM_PROVIDER,
    RAG_MIN_RELEVANCE_SCORE,
    RESULTS_DIR,
)
from app.generation.service import GenerationService, MockLLMProvider
from app.ingestion.loaders import load_questions_jsonl
from app.rag.service import RAGService


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = _REPO_ROOT / "data" / "evaluation" / "questions.jsonl"
OUTPUT_PATH = RESULTS_DIR / "citation_verification_eval.json"
SEP = "=" * 60


def compute_distribution(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics and percentiles for a list of numeric values."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p99": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)

    def percentile(p: float) -> float:
        idx = int(round(p * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return sorted_vals[idx]

    return {
        "count": n,
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "p10": round(percentile(0.10), 4),
        "p25": round(percentile(0.25), 4),
        "p50": round(percentile(0.50), 4),
        "p75": round(percentile(0.75), 4),
        "p90": round(percentile(0.90), 4),
        "p99": round(percentile(0.99), 4),
    }


def evaluate(include_citations: bool = True) -> int:
    print(f"\n{SEP}")
    print("PHASE 7: CITATION VERIFICATION & ANSWER GROUNDEDNESS EVALUATION")
    print(f"{SEP}\n")


    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found at: {QUESTIONS_PATH}")
        return 1

    # Initialize RAG Service with Citation Verification and optional citation generation
    try:
        citation_service = CitationVerificationService(
            min_overlap_threshold=CITATION_MIN_OVERLAP_THRESHOLD,
            entity_match_threshold=CITATION_ENTITY_MATCH_THRESHOLD,
        )
        if LLM_PROVIDER.lower() == "mock":
            gen_provider = MockLLMProvider(include_citations=include_citations)
            gen_service = GenerationService(provider=gen_provider)
        else:
            gen_service = GenerationService()

        rag_service = RAGService(
            generation_service=gen_service,
            citation_service=citation_service,
        )
        rag_service.initialize()
    except Exception as exc:
        print(f"ERROR: Failed to initialize RAG and Citation services: {exc}")
        return 1

    provider_name = rag_service.generation_service.provider.model_name
    is_live_llm = bool(os.getenv("OPENAI_API_KEY")) and LLM_PROVIDER.lower() == "openai"

    # Load evaluation questions
    questions = list(load_questions_jsonl(QUESTIONS_PATH))
    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"Configured LLM Provider: '{provider_name}' (Live LLM: {is_live_llm}, Citation Mode: {include_citations}).")
    print(f"Overlap Threshold: {CITATION_MIN_OVERLAP_THRESHOLD:.2f} | Entity Threshold: {CITATION_ENTITY_MATCH_THRESHOLD:.2f}")
    print("Running end-to-end RAG with Citation Verification (top_k=10)...\n")

    k_values = [1, 3, 5, 10]
    recalls: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_list: List[float] = []

    answerable_count = 0
    unanswerable_count = 0

    # Groundedness & Citation Metrics Accumulators
    total_evaluated_claims = 0
    total_grounded_claims = 0
    total_ungrounded_claims = 0
    fully_grounded_answers_count = 0

    total_citations_count = 0
    correct_citations_count = 0
    grounded_cited_claims_count = 0

    # Failure Mode Diagnostics
    hallucinated_claims_total = 0
    unsupported_claims_total = 0
    wrong_citations_total = 0
    missing_citations_total = 0
    invalid_citations_total = 0
    spurious_citations_total = 0

    # Unanswerable Question Accounting
    unanswerable_total = 0
    unanswerable_correct_refusals = 0
    unanswerable_fabricated = 0
    unanswerable_errors = 0

    # Raw score distributions
    all_overlap_scores: List[float] = []
    all_support_scores: List[float] = []

    detailed_results: List[Dict[str, Any]] = []

    for q in questions:
        # Perform retrieval to compute baseline retrieval metrics
        try:
            retrieval_resp = rag_service.hybrid_reranker.retrieve(q.question, top_k=10)
            retrieved_chunk_ids = [r.chunk_id for r in retrieval_resp.results]
            top_relevance_score = (
                retrieval_resp.results[0].similarity_score if retrieval_resp.results else None
            )
        except Exception as exc:
            retrieved_chunk_ids = []
            top_relevance_score = None

        # Execute end-to-end RAG with verification
        try:
            rag_response = rag_service.answer(q.question, top_k=10)
            execution_error = None
        except Exception as exc:
            execution_error = str(exc)
            logger.error(f"Error evaluating question '{q.question_id}': {exc}")
            rag_response = None

        qtype_str = (
            q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
        )
        target_chunk_ids = set(q.relevant_chunks)

        verification = rag_response.verification if rag_response else None

        # Extract per-claim score distributions
        if verification and verification.claims:
            for cv in verification.claims:
                if cv.claim.is_factual:
                    all_support_scores.append(cv.support_score)
                    token_overlap = cv.signals.get("token_precision", cv.support_score)
                    all_overlap_scores.append(token_overlap)

        q_detail: Dict[str, Any] = {
            "question_id": q.question_id,
            "question_type": qtype_str,
            "answerable": q.answerable,
            "relevant_chunks": q.relevant_chunks,
            "retrieved_chunks": retrieved_chunk_ids,
            "top_relevance_score": top_relevance_score,
            "generated_answer": rag_response.answer if rag_response else None,
            "verification": verification.model_dump() if verification else None,
            "error": execution_error,
        }

        # Unanswerable questions handling
        if not q.answerable or not target_chunk_ids:
            unanswerable_count += 1
            unanswerable_total += 1

            if execution_error is not None:
                unanswerable_errors += 1
            elif verification and verification.answer_type == AnswerType.INSUFFICIENT_EVIDENCE:
                unanswerable_correct_refusals += 1
                spurious_citations_total += verification.spurious_citations_count
            else:
                unanswerable_fabricated += 1

            detailed_results.append(q_detail)
            continue

        # Answerable questions handling
        answerable_count += 1

        # Retrieval metrics
        q_recalls: Dict[int, float] = {}
        for k in k_values:
            top_k_retrieved = set(retrieved_chunk_ids[:k])
            hit = 1.0 if bool(top_k_retrieved & target_chunk_ids) else 0.0
            q_recalls[k] = hit
            recalls[k].append(hit)

        first_rank = 0
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in target_chunk_ids:
                first_rank = rank
                break

        rr = (1.0 / first_rank) if first_rank > 0 else 0.0
        mrr_list.append(rr)

        q_detail["first_relevant_rank"] = first_rank if first_rank > 0 else None
        q_detail["reciprocal_rank"] = rr
        q_detail["recall"] = q_recalls

        # Groundedness & Citation aggregation
        if verification:
            total_evaluated_claims += verification.total_claims
            total_grounded_claims += verification.grounded_claims_count
            total_ungrounded_claims += verification.ungrounded_claims_count
            if verification.is_fully_grounded:
                fully_grounded_answers_count += 1

            total_citations_count += verification.total_citations
            correct_citations_count += verification.correct_citations_count

            hallucinated_claims_total += verification.hallucinated_claims_count
            unsupported_claims_total += verification.unsupported_claims_count
            wrong_citations_total += verification.wrong_citations_count
            missing_citations_total += verification.missing_citations_count
            invalid_citations_total += verification.invalid_citations_count
            spurious_citations_total += verification.spurious_citations_count

            for cv in verification.claims:
                if cv.claim.is_factual and cv.is_grounded and cv.has_citation and cv.is_citation_correct:
                    grounded_cited_claims_count += 1

        detailed_results.append(q_detail)

    # Compute macro-metrics
    avg_recalls = {
        k: (sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0 for k in k_values
    }
    avg_mrr = (sum(mrr_list) / len(mrr_list)) if mrr_list else 0.0

    claim_groundedness_rate = (
        (total_grounded_claims / total_evaluated_claims) if total_evaluated_claims > 0 else 1.0
    )
    fully_grounded_rate = (
        (fully_grounded_answers_count / answerable_count) if answerable_count > 0 else 1.0
    )

    if total_citations_count == 0:
        citation_precision: Optional[float] = None
        citation_f1: Optional[float] = None
    else:
        citation_precision = correct_citations_count / total_citations_count
        citation_recall_val = (
            (grounded_cited_claims_count / total_grounded_claims) if total_grounded_claims > 0 else 1.0
        )
        if (citation_precision + citation_recall_val) > 0.0:
            citation_f1 = (2.0 * citation_precision * citation_recall_val) / (
                citation_precision + citation_recall_val
            )
        else:
            citation_f1 = 0.0

    citation_recall = (
        (grounded_cited_claims_count / total_grounded_claims) if total_grounded_claims > 0 else 1.0
    )
    incorrect_citations_total = wrong_citations_total + invalid_citations_total

    correct_refusal_rate = (
        (unanswerable_correct_refusals / unanswerable_total) if unanswerable_total > 0 else 1.0
    )

    support_score_dist = compute_distribution(all_support_scores)
    overlap_score_dist = compute_distribution(all_overlap_scores)

    # Print Summary
    print(f"{SEP}")
    print("PHASE 7 EVALUATION SUMMARY")
    print(f"{SEP}\n")
    print(f"Total questions:          {len(questions)}")
    print(f"Answerable questions:     {answerable_count}")
    print(f"Unanswerable questions:   {unanswerable_count}\n")

    print("Retrieval Baseline Performance (Phase 5):")
    print(f"  Recall@1:     {avg_recalls[1]:.4f}  ({avg_recalls[1]*100:.2f}%)")
    print(f"  Recall@3:     {avg_recalls[3]:.4f}  ({avg_recalls[3]*100:.2f}%)")
    print(f"  Recall@5:     {avg_recalls[5]:.4f}  ({avg_recalls[5]*100:.2f}%)")
    print(f"  Recall@10:    {avg_recalls[10]:.4f}  ({avg_recalls[10]*100:.2f}%)")
    print(f"  MRR:          {avg_mrr:.4f}\n")

    print("Claim Groundedness Performance:")
    print(f"  Total Factual Claims Evaluated:  {total_evaluated_claims}")
    print(f"  Grounded Claims:                 {total_grounded_claims}")
    print(f"  Ungrounded Claims:               {total_ungrounded_claims}")
    print(f"  Claim Groundedness Rate:         {claim_groundedness_rate:.4f}  ({claim_groundedness_rate*100:.2f}%)")
    print(f"  Fully Grounded Answers Rate:     {fully_grounded_rate:.4f}  ({fully_grounded_rate*100:.2f}%)\n")

    print("Citation Attribution Performance:")
    print(f"  Citations Generated:             {total_citations_count}")
    print(f"  Correct Citations:               {correct_citations_count}")
    print(f"  Incorrect Citations:             {incorrect_citations_total}")
    if citation_precision is None:
        print("  Citation Precision:              N/A (0 citations generated)")
    else:
        print(f"  Citation Precision:              {citation_precision:.4f}  ({citation_precision*100:.2f}%)")
    print(f"  Citation Recall / Completeness:  {citation_recall:.4f}  ({citation_recall*100:.2f}%)")
    if citation_f1 is None:
        print("  Citation F1:                     N/A")
    else:
        print(f"  Citation F1:                     {citation_f1:.4f}\n")

    print("Unanswerable Questions & Refusal Accounting:")
    print(f"  Total Unanswerable:              {unanswerable_total}")
    print(f"  Correct Refusals:                {unanswerable_correct_refusals}")
    print(f"  Fabricated Answers:              {unanswerable_fabricated}")
    print(f"  Errors:                          {unanswerable_errors}")
    print(f"  Correct Refusal Rate:            {correct_refusal_rate:.4f}  ({correct_refusal_rate*100:.2f}%)\n")

    print("Failure Mode Diagnostics:")
    print(f"  Hallucinated Claims (Entity/Temp): {hallucinated_claims_total}")
    print(f"  Unsupported Claims:                {unsupported_claims_total}")
    print(f"  Wrong Citations:                   {wrong_citations_total}")
    print(f"  Missing Citations:                 {missing_citations_total}")
    print(f"  Invalid Citation Indices:          {invalid_citations_total}")
    print(f"  Spurious Citations:                {spurious_citations_total}\n")

    print("Score Distributions (Claim Support Scores):")
    print(f"  Mean: {support_score_dist['mean']:.4f} ± {support_score_dist['std']:.4f}")
    print(f"  Min:  {support_score_dist['min']:.4f} | Max: {support_score_dist['max']:.4f}")
    print(f"  P25:  {support_score_dist['p25']:.4f} | P50: {support_score_dist['p50']:.4f} | P75: {support_score_dist['p75']:.4f} | P90: {support_score_dist['p90']:.4f}\n")

    output_data = {
        "pipeline": "RAG (Reranked Hybrid Retrieval + LLM Generation + Citation Verification)",
        "llm_provider": provider_name,
        "is_live_llm_evaluated": is_live_llm,
        "citation_generation_mode": include_citations,
        "total_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "configuration": {
            "min_overlap_threshold": CITATION_MIN_OVERLAP_THRESHOLD,
            "entity_match_threshold": CITATION_ENTITY_MATCH_THRESHOLD,
            "min_relevance_threshold": rag_service.min_relevance_score,
        },
        "retrieval_metrics": {
            "recall_at_1": avg_recalls[1],
            "recall_at_3": avg_recalls[3],
            "recall_at_5": avg_recalls[5],
            "recall_at_10": avg_recalls[10],
            "mrr": avg_mrr,
        },
        "groundedness_metrics": {
            "total_evaluated_claims": total_evaluated_claims,
            "grounded_claims_count": total_grounded_claims,
            "ungrounded_claims_count": total_ungrounded_claims,
            "claim_groundedness_rate": round(claim_groundedness_rate, 4),
            "fully_grounded_answers_rate": round(fully_grounded_rate, 4),
        },
        "citation_metrics": {
            "citations_generated": total_citations_count,
            "correct_citations_count": correct_citations_count,
            "incorrect_citations_count": incorrect_citations_total,
            "citation_precision": round(citation_precision, 4) if citation_precision is not None else None,
            "citation_completeness": round(citation_recall, 4),
            "citation_f1": round(citation_f1, 4) if citation_f1 is not None else None,
        },
        "unanswerable_metrics": {
            "unanswerable_total": unanswerable_total,
            "correct_refusals": unanswerable_correct_refusals,
            "fabricated_answers": unanswerable_fabricated,
            "errors": unanswerable_errors,
            "correct_refusal_rate": round(correct_refusal_rate, 4),
        },
        "failure_mode_diagnostics": {
            "hallucinated_claims_total": hallucinated_claims_total,
            "unsupported_claims_total": unsupported_claims_total,
            "wrong_citations_total": wrong_citations_total,
            "missing_citations_total": missing_citations_total,
            "invalid_citations_total": invalid_citations_total,
            "spurious_citations_total": spurious_citations_total,
        },
        "score_distributions": {
            "support_scores": support_score_dist,
            "overlap_scores": overlap_score_dist,
        },
        "detailed_results": detailed_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2)

    print(f"Saved machine-readable evaluation results to:")
    print(f"  {OUTPUT_PATH.relative_to(_REPO_ROOT)}")

    print(f"\n{SEP}")
    print("PHASE 7 CITATION VERIFICATION EVALUATION PASSED")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Citation Verification and Groundedness"
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Run without attaching citation markers in mock generation",
    )
    args = parser.parse_args()
    sys.exit(evaluate(include_citations=not args.no_citations))

