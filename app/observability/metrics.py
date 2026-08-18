"""Phase 9 in-memory metrics collector with thread-safe counters and latency percentiles."""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional


def _compute_percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute summary statistics and percentiles safely.

    Handles empty sample lists without division by zero.
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = round(sum(sorted_vals) / n, 2)
    min_val = round(sorted_vals[0], 2)
    max_val = round(sorted_vals[-1], 2)

    def _get_percentile(p: float) -> float:
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
        "mean": mean_val,
        "min": min_val,
        "max": max_val,
        "p50": _get_percentile(0.50),
        "p90": _get_percentile(0.90),
        "p95": _get_percentile(0.95),
        "p99": _get_percentile(0.99),
    }


class MetricsCollector:
    """Thread-safe application metrics accumulator."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "query_requests": 0,
            "retrieve_requests": 0,
            "health_requests": 0,
            "ready_requests": 0,
            "metrics_requests": 0,
            "validation_errors": 0,
            "internal_errors": 0,
            "not_found_errors": 0,
            "method_not_allowed_errors": 0,
            "retrieval_executions": 0,
            "evidence_sufficient": 0,
            "evidence_insufficient": 0,
            "llm_executions": 0,
            "llm_skipped_due_to_insufficient_evidence": 0,
            "generation_errors": 0,
            "citation_verifications": 0,
            "grounded_answers": 0,
            "ungrounded_answers": 0,
        }
        self._latencies: Dict[str, List[float]] = {
            "total_request": [],
            "retrieval": [],
            "generation": [],
            "verification": [],
        }

    def increment(self, metric_name: str, amount: int = 1) -> None:
        """Increment a named integer counter."""
        with self._lock:
            self._counters[metric_name] = self._counters.get(metric_name, 0) + amount

    def record_latency(self, metric_name: str, latency_ms: float) -> None:
        """Record a latency measurement sample in milliseconds."""
        if latency_ms is None or latency_ms < 0:
            return
        with self._lock:
            if metric_name not in self._latencies:
                self._latencies[metric_name] = []
            self._latencies[metric_name].append(float(latency_ms))

    def get_snapshot(self) -> Dict[str, Any]:
        """Return an immutable snapshot of all metrics."""
        with self._lock:
            counters_copy = dict(self._counters)
            latencies_copy = {k: list(v) for k, v in self._latencies.items()}

        latency_stats = {
            k: _compute_percentiles(v) for k, v in latencies_copy.items()
        }

        # Safe error rate calculation
        total = counters_copy.get("total_requests", 0)
        failed = counters_copy.get("failed_requests", 0)
        error_rate = round(failed / total, 4) if total > 0 else 0.0

        # Safe groundedness rate calculation
        total_verified = counters_copy.get("citation_verifications", 0)
        grounded = counters_copy.get("grounded_answers", 0)
        groundedness_rate = round(grounded / total_verified, 4) if total_verified > 0 else None

        errors_by_status = {
            k.replace("status_", ""): v
            for k, v in counters_copy.items()
            if k.startswith("status_") and int(k.replace("status_", "")) >= 400 and v > 0
        }

        return {
            "requests": {
                "total": counters_copy.get("total_requests", 0),
                "successful": counters_copy.get("successful_requests", 0),
                "failed": counters_copy.get("failed_requests", 0),
                "error_rate": error_rate,
                "by_endpoint": {
                    "/query": counters_copy.get("query_requests", 0),
                    "/retrieve": counters_copy.get("retrieve_requests", 0),
                    "/health": counters_copy.get("health_requests", 0),
                    "/ready": counters_copy.get("ready_requests", 0),
                    "/metrics": counters_copy.get("metrics_requests", 0),
                },
                "errors_by_status": errors_by_status,
                "errors_by_status_code": errors_by_status,
            },
            "errors": {
                "validation_errors": counters_copy.get("validation_errors", 0),
                "internal_errors": counters_copy.get("internal_errors", 0),
                "not_found_errors": counters_copy.get("not_found_errors", 0),
                "method_not_allowed_errors": counters_copy.get("method_not_allowed_errors", 0),
            },
            "rag": {
                "retrieval_executions": counters_copy.get("retrieval_executions", 0),
                "evidence_sufficient": counters_copy.get("evidence_sufficient", 0),
                "evidence_insufficient": counters_copy.get("evidence_insufficient", 0),
                "llm_executions": counters_copy.get("llm_executions", 0),
                "llm_skipped_due_to_insufficient_evidence": counters_copy.get(
                    "llm_skipped_due_to_insufficient_evidence", 0
                ),
                "generation_errors": counters_copy.get("generation_errors", 0),
                "citation_verifications": counters_copy.get("citation_verifications", 0),
                "grounded_answers": counters_copy.get("grounded_answers", 0),
                "ungrounded_answers": counters_copy.get("ungrounded_answers", 0),
                "groundedness_rate": groundedness_rate,
            },
            "latency_ms": latency_stats,
        }

    def reset(self) -> None:
        """Reset all counters and latency distributions (for tests)."""
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
            for k in self._latencies:
                self._latencies[k] = []


# Singleton metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Return singleton MetricsCollector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics() -> None:
    """Reset the singleton metrics collector."""
    global _metrics_collector
    if _metrics_collector is not None:
        _metrics_collector.reset()
    else:
        _metrics_collector = MetricsCollector()
