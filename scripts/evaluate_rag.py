#!/usr/bin/env python3
"""
scripts/evaluate_rag.py
=======================
CLI script to evaluate RAG Orchestration (Retrieval + Grounded LLM Generation + Evidence Sufficiency)
performance against the 150 evaluation questions in the Nexora dataset.

Calculates:
  - Retrieval Metrics: Recall@1, Recall@3, Recall@5, Recall@10, MRR
  - Generation Metrics: Total executions, insufficient evidence answers, errors
  - Unanswerable Metrics: Total, insufficient evidence responses, fabricated answers, errors

Saves machine-readable evaluation results to `results/rag_eval.json`.

Usage
-----
    python scripts/evaluate_rag.py

Exit codes
----------
    0 — Evaluation completed successfully.
    1 — Evaluation failed.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Path bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import (  # noqa: E402
    LLM_PROVIDER,
    RAG_MIN_RELEVANCE_SCORE,
    RESULTS_DIR,
)
from app.ingestion.loaders import load_questions_jsonl  # noqa: E402
from app.rag.service import RAGService  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = _REPO_ROOT / "data" / "evaluation" / "questions.jsonl"
OUTPUT_PATH = RESULTS_DIR / "rag_eval.json"
SEP = "=" * 40


def evaluate() -> int:
    print(f"\n{SEP}")
    print("RAG PIPELINE (RETRIEVAL + GENERATION + EVIDENCE SUFFICIENCY) EVALUATION")
    print(f"{SEP}\n")

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found at: {QUESTIONS_PATH}")
        return 1

    # Initialize RAG Service
    try:
        rag_service = RAGService()
        rag_service.initialize()
    except Exception as exc:
        print(f"ERROR: Failed to initialize RAG service: {exc}")
        return 1

    provider_name = rag_service.generation_service.provider.model_name
    is_live_llm = bool(os.getenv("OPENAI_API_KEY")) and LLM_PROVIDER.lower() == "openai"

    # Load evaluation questions
    questions = list(load_questions_jsonl(QUESTIONS_PATH))
    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"Configured LLM Provider: '{provider_name}' (Live LLM: {is_live_llm}).")
    print(f"Configured Evidence Min Relevance Threshold: {RAG_MIN_RELEVANCE_SCORE:.4f}")
    print("Running end-to-end RAG evaluation (top_k=10)...\n")

    k_values = [1, 3, 5, 10]
    recalls: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_list: List[float] = []

    answerable_count = 0
    unanswerable_count = 0

    retrieval_execution_count = 0
    evidence_sufficient_count = 0
    evidence_insufficient_count = 0
    llm_execution_count = 0
    llm_skipped_count = 0
    errors_count = 0

    # Unanswerable question breakdown
    unanswerable_total = 0
    unanswerable_insufficient = 0
    unanswerable_answered = 0
    unanswerable_errors = 0

    detailed_results: List[Dict[str, Any]] = []

    for q in questions:
        # Also perform retrieval to log exact top relevance score in detailed results
        try:
            retrieval_resp = rag_service.hybrid_reranker.retrieve(q.question, top_k=10)
            retrieval_execution_count += 1
            top_relevance_score = (
                retrieval_resp.results[0].similarity_score if retrieval_resp.results else None
            )
            retrieved_chunk_ids = [r.chunk_id for r in retrieval_resp.results]
        except Exception as exc:
            top_relevance_score = None
            retrieved_chunk_ids = []

        try:
            rag_response = rag_service.answer(q.question, top_k=10)
            execution_error = None
        except Exception as exc:
            errors_count += 1
            execution_error = str(exc)
            logger.error(f"Error evaluating question '{q.question_id}': {exc}")
            rag_response = None

        qtype_str = (
            q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
        )

        is_insufficient = (
            rag_response is not None and "sufficient information" in rag_response.answer.lower()
        )

        evidence_sufficient = (
            top_relevance_score is not None
            and top_relevance_score >= rag_service.min_relevance_score
        )

        if evidence_sufficient:
            evidence_sufficient_count += 1
            if rag_response is not None:
                llm_execution_count += 1
        else:
            evidence_insufficient_count += 1
            llm_skipped_count += 1

        q_detail: Dict[str, Any] = {
            "question_id": q.question_id,
            "question_type": qtype_str,
            "answerable": q.answerable,
            "relevant_chunks": q.relevant_chunks,
            "retrieved_chunks": retrieved_chunk_ids,
            "top_relevance_score": top_relevance_score,
            "evidence_sufficient_flag": evidence_sufficient,
            "generated_answer": rag_response.answer if rag_response else None,
            "insufficient_evidence_flag": is_insufficient,
            "llm_executed": evidence_sufficient and (rag_response is not None),
            "error": execution_error,
        }

        target_chunk_ids = set(q.relevant_chunks)

        if not q.answerable or not target_chunk_ids:
            unanswerable_count += 1
            unanswerable_total += 1
            if execution_error is not None:
                unanswerable_errors += 1
            elif is_insufficient:
                unanswerable_insufficient += 1
            else:
                unanswerable_answered += 1

            q_detail["retrieval_metrics"] = "unanswerable (n/a for chunk recall)"
            detailed_results.append(q_detail)
            continue

        answerable_count += 1

        # Retrieval Recall@K (Hit Rate) for answerable questions
        q_recalls: Dict[int, float] = {}
        for k in k_values:
            top_k_retrieved = set(retrieved_chunk_ids[:k])
            hit = 1.0 if bool(top_k_retrieved & target_chunk_ids) else 0.0
            q_recalls[k] = hit
            recalls[k].append(hit)

        # Reciprocal Rank
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
        detailed_results.append(q_detail)

    # Averages
    avg_recalls = {
        k: (sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0 for k in k_values
    }
    avg_mrr = (sum(mrr_list) / len(mrr_list)) if mrr_list else 0.0

    # Print summary
    print(f"{SEP}")
    print("RAG EVALUATION SUMMARY")
    print(f"{SEP}\n")
    print(f"Total questions:        {len(questions)}")
    print(f"Answerable questions:   {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}\n")

    print("Retrieval Component Performance:")
    print(f"  Recall@1:   {avg_recalls[1]:.4f}  ({avg_recalls[1]*100:.2f}%)")
    print(f"  Recall@3:   {avg_recalls[3]:.4f}  ({avg_recalls[3]*100:.2f}%)")
    print(f"  Recall@5:   {avg_recalls[5]:.4f}  ({avg_recalls[5]*100:.2f}%)")
    print(f"  Recall@10:  {avg_recalls[10]:.4f}  ({avg_recalls[10]*100:.2f}%)")
    print(f"  MRR:        {avg_mrr:.4f}\n")

    print("Generation & Evidence Accounting:")
    print(f"  Retrieval Executions:        {retrieval_execution_count}")
    print(f"  Evidence Sufficient:         {evidence_sufficient_count}")
    print(f"  Evidence Insufficient:       {evidence_insufficient_count}")
    print(f"  LLM Executions:              {llm_execution_count}")
    print(f"  LLM Skipped (Insufficient):  {llm_skipped_count}")
    print(f"  Execution Errors:            {errors_count}")
    print(f"  Live LLM API Evaluated:      {is_live_llm}\n")

    print("Unanswerable Questions Breakdown:")
    print(f"  Total Unanswerable:          {unanswerable_total}")
    print(f"  Insufficient Info Responses: {unanswerable_insufficient}")
    print(f"  Fabricated/Answered:         {unanswerable_answered}")
    print(f"  Errors:                      {unanswerable_errors}\n")

    if not is_live_llm:
        print("NOTE: Mock LLM Provider was used for offline evaluation.")
        print("      Semantic answer quality metrics (e.g. BERTScore/ROUGE) require a live LLM API.")

    # Save output JSON
    output_data = {
        "pipeline": "RAG (Reranked Hybrid Retrieval + Evidence Sufficiency + LLM Generation)",
        "llm_provider": provider_name,
        "is_live_llm_evaluated": is_live_llm,
        "total_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "min_relevance_threshold": rag_service.min_relevance_score,
        "retrieval_metrics": {
            "recall_at_1": avg_recalls[1],
            "recall_at_3": avg_recalls[3],
            "recall_at_5": avg_recalls[5],
            "recall_at_10": avg_recalls[10],
            "mrr": avg_mrr,
        },
        "generation_metrics": {
            "retrieval_executions": retrieval_execution_count,
            "evidence_sufficient_count": evidence_sufficient_count,
            "evidence_insufficient_count": evidence_insufficient_count,
            "llm_execution_count": llm_execution_count,
            "llm_skipped_count": llm_skipped_count,
            "errors_count": errors_count,
            "provider_name": provider_name,
        },
        "unanswerable_metrics": {
            "unanswerable_total": unanswerable_total,
            "unanswerable_insufficient": unanswerable_insufficient,
            "unanswerable_answered": unanswerable_answered,
            "unanswerable_errors": unanswerable_errors,
        },
        "detailed_results": detailed_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2)

    print(f"\nSaved machine-readable evaluation results to:")
    print(f"  {OUTPUT_PATH.relative_to(_REPO_ROOT)}")

    print(f"\n{SEP}")
    print("RAG EVALUATION PASSED")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate())
