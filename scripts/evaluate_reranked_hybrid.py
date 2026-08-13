#!/usr/bin/env python3
"""
scripts/evaluate_reranked_hybrid.py
====================================
CLI script to evaluate Reranked Hybrid Retrieval (Hybrid RRF + Cross-Encoder Reranker)
performance against the 150 evaluation questions in the Nexora dataset.

Calculates:
  - Recall@1
  - Recall@3
  - Recall@5
  - Recall@10
  - Mean Reciprocal Rank (MRR)

Saves machine-readable results to `results/reranked_hybrid_eval.json`.

Usage
-----
    python scripts/evaluate_reranked_hybrid.py

Exit codes
----------
    0 — Evaluation completed successfully.
    1 — Evaluation failed.
"""

from __future__ import annotations

import io
import json
import logging
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
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RRF_K,
    RERANKER_MODEL_NAME,
    RESULTS_DIR,
)
from app.ingestion.loaders import load_questions_jsonl  # noqa: E402
from app.ingestion.models import QuestionType  # noqa: E402
from app.reranking.hybrid_reranker import RerankedHybridService  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = _REPO_ROOT / "data" / "evaluation" / "questions.jsonl"
OUTPUT_PATH = RESULTS_DIR / "reranked_hybrid_eval.json"
SEP = "=" * 40


def evaluate() -> int:
    print(f"\n{SEP}")
    print("RERANKED HYBRID RETRIEVAL (CROSS-ENCODER) EVALUATION")
    print(f"{SEP}\n")

    if not QUESTIONS_PATH.exists():
        print(f"ERROR: Questions file not found at: {QUESTIONS_PATH}")
        return 1

    # Initialize Reranked Hybrid Retrieval Service
    try:
        service = RerankedHybridService()
        service.initialize()
    except Exception as exc:
        print(f"ERROR: Failed to initialize reranked hybrid retrieval service: {exc}")
        print("Ensure vector index is built using `python scripts/build_vector_index.py`.")
        return 1

    # Load evaluation questions
    questions = list(load_questions_jsonl(QUESTIONS_PATH))
    print(f"Loaded {len(questions)} evaluation questions.")
    print(
        f"Running Reranked Hybrid retrieval (top_k=10, model='{RERANKER_MODEL_NAME}', candidate_k={DEFAULT_RERANK_CANDIDATE_K}, rrf_k={DEFAULT_RRF_K})...\n"
    )

    # Evaluation accumulators
    k_values = [1, 3, 5, 10]
    recalls: Dict[int, List[float]] = {k: [] for k in k_values}
    mrr_list: List[float] = []

    # Category breakdown
    category_metrics: Dict[str, Dict[str, Any]] = {}
    detailed_results: List[Dict[str, Any]] = []

    answerable_count = 0
    unanswerable_count = 0

    for q in questions:
        response = service.retrieve(
            q.question,
            top_k=10,
            rrf_k=DEFAULT_RRF_K,
            candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        )
        retrieved_chunk_ids = [res.chunk_id for res in response.results]
        target_chunk_ids = set(q.relevant_chunks)

        qtype_str = (
            q.question_type.value if hasattr(q.question_type, "value") else str(q.question_type)
        )

        q_detail: Dict[str, Any] = {
            "question_id": q.question_id,
            "question_type": qtype_str,
            "answerable": q.answerable,
            "relevant_chunks": q.relevant_chunks,
            "retrieved_chunks": retrieved_chunk_ids,
        }

        if not q.answerable or not target_chunk_ids:
            unanswerable_count += 1
            q_detail["metrics"] = "unanswerable (n/a for chunk recall)"
            detailed_results.append(q_detail)
            continue

        answerable_count += 1

        # Calculate Recall@K (Hit Rate) for answerable questions
        q_recalls: Dict[int, float] = {}
        for k in k_values:
            top_k_retrieved = set(retrieved_chunk_ids[:k])
            hit = 1.0 if bool(top_k_retrieved & target_chunk_ids) else 0.0
            q_recalls[k] = hit
            recalls[k].append(hit)

        # Calculate Reciprocal Rank
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

        # Track per-category metrics
        if qtype_str not in category_metrics:
            category_metrics[qtype_str] = {
                "total": 0,
                "answerable": 0,
                "recalls": {k: [] for k in k_values},
                "mrr": [],
            }
        cat = category_metrics[qtype_str]
        cat["total"] += 1
        cat["answerable"] += 1
        for k in k_values:
            cat["recalls"][k].append(q_recalls[k])
        cat["mrr"].append(rr)

    # Compute overall metric averages
    avg_recalls = {
        k: (sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0 for k in k_values
    }
    avg_mrr = (sum(mrr_list) / len(mrr_list)) if mrr_list else 0.0

    # Print clean evaluation report
    print(f"{SEP}")
    print("RERANKED HYBRID EVALUATION RESULTS")
    print(f"{SEP}\n")
    print(f"Questions evaluated:  {len(questions)}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable:         {unanswerable_count}\n")

    print(f"Recall@1:   {avg_recalls[1]:.4f}  ({avg_recalls[1]*100:.2f}%)")
    print(f"Recall@3:   {avg_recalls[3]:.4f}  ({avg_recalls[3]*100:.2f}%)")
    print(f"Recall@5:   {avg_recalls[5]:.4f}  ({avg_recalls[5]*100:.2f}%)")
    print(f"Recall@10:  {avg_recalls[10]:.4f}  ({avg_recalls[10]*100:.2f}%)\n")

    print(f"MRR:        {avg_mrr:.4f}\n")

    print("Category Breakdown (Recall@5 & MRR):")
    print("-" * 45)
    print(f"{'Category':<18} {'Count':<6} {'R@5':<8} {'MRR':<8}")
    print("-" * 45)

    category_summary: Dict[str, Any] = {}
    for qtype in [qt.value for qt in QuestionType]:
        if qtype in category_metrics:
            cat = category_metrics[qtype]
            c_r5 = (sum(cat["recalls"][5]) / len(cat["recalls"][5])) if cat["recalls"][5] else 0.0
            c_mrr = (sum(cat["mrr"]) / len(cat["mrr"])) if cat["mrr"] else 0.0
            print(f"{qtype:<18} {cat['total']:<6} {c_r5:<8.4f} {c_mrr:<8.4f}")
            category_summary[qtype] = {
                "count": cat["total"],
                "recall_at_1": (
                    (sum(cat["recalls"][1]) / len(cat["recalls"][1])) if cat["recalls"][1] else 0.0
                ),
                "recall_at_3": (
                    (sum(cat["recalls"][3]) / len(cat["recalls"][3])) if cat["recalls"][3] else 0.0
                ),
                "recall_at_5": c_r5,
                "recall_at_10": (
                    (sum(cat["recalls"][10]) / len(cat["recalls"][10])) if cat["recalls"][10] else 0.0
                ),
                "mrr": c_mrr,
            }
        elif qtype == QuestionType.UNANSWERABLE.value:
            print(f"{qtype:<18} {unanswerable_count:<6} {'N/A':<8} {'N/A':<8}")
            category_summary[qtype] = {
                "count": unanswerable_count,
                "note": "Unanswerable questions have no relevant chunks in corpus.",
            }

    print("-" * 45)

    # Save output JSON
    output_data = {
        "retrieval_mode": "reranked_hybrid",
        "reranker_model": RERANKER_MODEL_NAME,
        "rrf_k": DEFAULT_RRF_K,
        "candidate_k": DEFAULT_RERANK_CANDIDATE_K,
        "total_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "metrics": {
            "recall_at_1": avg_recalls[1],
            "recall_at_3": avg_recalls[3],
            "recall_at_5": avg_recalls[5],
            "recall_at_10": avg_recalls[10],
            "mrr": avg_mrr,
        },
        "category_breakdown": category_summary,
        "detailed_results": detailed_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2)

    print(f"\nSaved machine-readable evaluation results to:")
    print(f"  {OUTPUT_PATH.relative_to(_REPO_ROOT)}")

    print(f"\n{SEP}")
    print("RERANKED HYBRID EVALUATION PASSED")
    print(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate())
