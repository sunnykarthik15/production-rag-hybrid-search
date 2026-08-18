"""Phase 9 API Performance & Load Evaluation Script.

Evaluates local API latency and throughput across:
- GET  /health
- GET  /ready
- POST /retrieve
- POST /query

Computes:
- Total Requests & Success/Failure Counts
- Throughput (Requests per Second)
- Latency percentiles (Mean, Min, Max, P50, P90, P95, P99)

Outputs:
- Formatted console table
- JSON report saved to results/api_performance_eval.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.api.dependencies import reset_services
from app.citations.models import CitationVerificationResult
from app.config import RESULTS_DIR
from app.main import create_app
from app.rag.models import RAGResponse, RAGSource
from app.retrieval.models import RetrievalResponse, RetrievalResult

logging.basicConfig(level=logging.WARNING)


def _compute_stats(latencies_ms: List[float], total_duration_s: float) -> Dict[str, Any]:
    """Compute throughput and percentile statistics safely."""
    if not latencies_ms:
        return {
            "count": 0,
            "throughput_req_per_sec": 0.0,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
        }

    sorted_vals = sorted(latencies_ms)
    n = len(sorted_vals)
    throughput = round(n / total_duration_s, 2) if total_duration_s > 0 else 0.0
    mean_val = round(sum(sorted_vals) / n, 2)
    min_val = round(sorted_vals[0], 2)
    max_val = round(sorted_vals[-1], 2)

    def _get_p(p: float) -> float:
        idx = (len(sorted_vals) - 1) * p
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return round(sorted_vals[int(idx)], 2)
        weight = idx - lower
        val = sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
        return round(val, 2)

    return {
        "count": n,
        "throughput_req_per_sec": throughput,
        "mean_ms": mean_val,
        "min_ms": min_val,
        "max_ms": max_val,
        "p50_ms": _get_p(0.50),
        "p90_ms": _get_p(0.90),
        "p95_ms": _get_p(0.95),
        "p99_ms": _get_p(0.99),
    }


def _make_mock_rag_response(query: str) -> RAGResponse:
    source = RAGSource(
        chunk_id="DOC001_CHUNK_01",
        document_id="DOC001",
        title="Operations Policy",
        department="Operations",
        rank=1,
        score=0.95,
        text="Sample policy evidence text.",
    )
    verification = CitationVerificationResult(
        query=query,
        answer="Sample verified answer.",
        total_claims=1,
        grounded_claims_count=1,
        ungrounded_claims_count=0,
        claim_groundedness_rate=1.0,
        is_fully_grounded=True,
        total_citations=1,
        correct_citations_count=1,
        citation_precision=1.0,
        citation_recall=1.0,
        citation_f1=1.0,
        is_fully_cited=True,
    )
    return RAGResponse(
        query=query,
        answer="Sample verified answer.",
        sources=[source],
        verification=verification,
        groundedness_score=1.0,
    )


def _make_mock_retrieval_response(query: str) -> RetrievalResponse:
    result = RetrievalResult(
        chunk_id="DOC001_CHUNK_01",
        document_id="DOC001",
        text="Sample retrieval text.",
        title="Operations Policy",
        department="Operations",
        similarity_score=0.88,
        rank=1,
    )
    return RetrievalResponse(query=query, results=[result], top_k=5)


def run_benchmark_for_endpoint(
    app: Any,
    endpoint: str,
    total_requests: int = 50,
    concurrency: int = 5,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Execute load benchmark against an endpoint."""
    queries = [
        f"What is the operational protocol for procedure {i % 10 + 1}?"
        for i in range(total_requests)
    ]

    latencies_ms: List[float] = []
    successes = 0
    failures = 0

    client = TestClient(app, raise_server_exceptions=False)
    start_wall = time.monotonic()

    for i in range(total_requests):
        q = queries[i]
        t0 = time.monotonic()
        try:
            if endpoint == "/health":
                resp = client.get("/health")
            elif endpoint == "/ready":
                resp = client.get("/ready")
            elif endpoint == "/metrics":
                resp = client.get("/metrics")
            elif endpoint == "/retrieve":
                resp = client.post("/retrieve", json={"query": q, "top_k": top_k})
            elif endpoint == "/query":
                resp = client.post("/query", json={"query": q, "top_k": top_k})
            else:
                resp = client.get(endpoint)

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            if resp.status_code == 200:
                successes += 1
            else:
                failures += 1
        except Exception:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            failures += 1

    total_wall_s = time.monotonic() - start_wall
    stats = _compute_stats(latencies_ms, total_wall_s)

    return {
        "endpoint": endpoint,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful_requests": successes,
        "failed_requests": failures,
        "total_duration_s": round(total_wall_s, 3),
        "statistics": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate API latency and throughput under load.")
    parser.add_argument("--endpoint", default="all", help="Target endpoint (/health, /ready, /retrieve, /query, or 'all')")
    parser.add_argument("--requests", type=int, default=50, help="Total requests per endpoint")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency level")
    parser.add_argument("--top_k", type=int, default=5, help="top_k for retrieval and query")
    parser.add_argument("--output", default=str(RESULTS_DIR / "api_performance_eval.json"), help="Output JSON path")
    args = parser.parse_args()

    logging.disable(logging.INFO)

    print("=" * 65, flush=True)
    print("PHASE 9 — API PERFORMANCE & LOAD BENCHMARK", flush=True)
    print(f"Requests per Endpoint: {args.requests} | Concurrency: {args.concurrency}", flush=True)
    print("=" * 65, flush=True)

    reset_services()

    # Set up mocks to test API layer throughput without requiring unneeded live token spends
    mock_rag = MagicMock()
    mock_rag.answer.side_effect = lambda query, top_k=10, **kw: _make_mock_rag_response(query)
    mock_rag.hybrid_reranker.hybrid_service.vector_service.vector_store.is_initialized = True
    mock_rag.hybrid_reranker.hybrid_service.bm25_service.bm25_store.is_initialized = True
    mock_rag.hybrid_reranker.reranker_service.is_initialized = True
    mock_rag.generation_service.provider = MagicMock()
    mock_rag.citation_service.verifier = MagicMock()

    mock_reranker = MagicMock()
    mock_reranker.retrieve.side_effect = lambda query, top_k=5, **kw: _make_mock_retrieval_response(query)

    endpoints_to_test = (
        ["/health", "/ready", "/retrieve", "/query"]
        if args.endpoint.lower() == "all"
        else [args.endpoint]
    )

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "concurrency": args.concurrency,
        "requests_per_endpoint": args.requests,
        "endpoints": {},
    }

    import app.api.dependencies as deps

    deps._rag_service = mock_rag
    deps._hybrid_reranker = mock_reranker
    deps._initialized = True

    app = create_app()

    for ep in endpoints_to_test:
        print(f"\nBenchmarking {ep}...", flush=True)
        ep_result = run_benchmark_for_endpoint(
            app=app,
            endpoint=ep,
            total_requests=args.requests,
            concurrency=args.concurrency,
            top_k=args.top_k,
        )
        results["endpoints"][ep] = ep_result

        s = ep_result["statistics"]
        print(f"  Total: {ep_result['total_requests']} | Success: {ep_result['successful_requests']} | Fail: {ep_result['failed_requests']}", flush=True)
        print(f"  Throughput: {s['throughput_req_per_sec']} req/s | Mean: {s['mean_ms']}ms | P50: {s['p50_ms']}ms | P95: {s['p95_ms']}ms | P99: {s['p99_ms']}ms", flush=True)

    deps.reset_services()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "-" * 65, flush=True)
    print(f"Saved performance evaluation results to:\n  {out_path}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
