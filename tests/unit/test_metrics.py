"""Unit tests for Phase 9 MetricsCollector and percentile statistics."""

from __future__ import annotations

import concurrent.futures

from app.observability.metrics import (
    MetricsCollector,
    _compute_percentiles,
    get_metrics_collector,
    reset_metrics,
)


class TestPercentileCalculations:
    """Test percentile and statistical summary calculations."""

    def test_empty_samples_safe(self):
        """Empty sample list returns count=0 and None for all percentiles without zero division."""
        stats = _compute_percentiles([])
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["p50"] is None
        assert stats["p95"] is None
        assert stats["p99"] is None

    def test_single_sample(self):
        """Single sample sets all percentiles to that value."""
        stats = _compute_percentiles([100.0])
        assert stats["count"] == 1
        assert stats["mean"] == 100.0
        assert stats["min"] == 100.0
        assert stats["max"] == 100.0
        assert stats["p50"] == 100.0
        assert stats["p95"] == 100.0
        assert stats["p99"] == 100.0

    def test_multiple_samples_percentiles(self):
        """Multiple samples compute mathematically correct percentiles."""
        samples = [float(i) for i in range(1, 101)]  # 1.0 to 100.0
        stats = _compute_percentiles(samples)
        assert stats["count"] == 100
        assert stats["mean"] == 50.5
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["p50"] == 50.5
        assert stats["p90"] == 90.1
        assert stats["p95"] == 95.05
        assert stats["p99"] == 99.01


class TestMetricsCollector:
    """Test MetricsCollector counter tracking, snapshot generation, and thread safety."""

    def test_initial_state_zero(self):
        """Fresh MetricsCollector starts with zero counts and empty latencies."""
        collector = MetricsCollector()
        snap = collector.get_snapshot()
        assert snap["requests"]["total"] == 0
        assert snap["requests"]["successful"] == 0
        assert snap["requests"]["failed"] == 0
        assert snap["requests"]["error_rate"] == 0.0
        assert snap["rag"]["groundedness_rate"] is None

    def test_increment_and_error_rate(self):
        """Incrementing requests updates counters and computes error rates."""
        collector = MetricsCollector()
        collector.increment("total_requests", 10)
        collector.increment("successful_requests", 8)
        collector.increment("failed_requests", 2)
        collector.increment("query_requests", 6)
        collector.increment("retrieve_requests", 4)

        snap = collector.get_snapshot()
        assert snap["requests"]["total"] == 10
        assert snap["requests"]["successful"] == 8
        assert snap["requests"]["failed"] == 2
        assert snap["requests"]["error_rate"] == 0.2
        assert snap["requests"]["by_endpoint"]["/query"] == 6
        assert snap["requests"]["by_endpoint"]["/retrieve"] == 4

    def test_groundedness_rate_calculation(self):
        """Groundedness rate is computed safely when verifications exist."""
        collector = MetricsCollector()
        collector.increment("citation_verifications", 5)
        collector.increment("grounded_answers", 4)
        collector.increment("ungrounded_answers", 1)

        snap = collector.get_snapshot()
        assert snap["rag"]["citation_verifications"] == 5
        assert snap["rag"]["grounded_answers"] == 4
        assert snap["rag"]["ungrounded_answers"] == 1
        assert snap["rag"]["groundedness_rate"] == 0.8

    def test_record_latency_distribution(self):
        """Recorded latencies populate distribution stats."""
        collector = MetricsCollector()
        collector.record_latency("total_request", 10.0)
        collector.record_latency("total_request", 20.0)
        collector.record_latency("total_request", 30.0)

        snap = collector.get_snapshot()
        stats = snap["latency_ms"]["total_request"]
        assert stats["count"] == 3
        assert stats["mean"] == 20.0
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0

    def test_negative_or_none_latency_ignored(self):
        """Negative or None latencies are safely ignored."""
        collector = MetricsCollector()
        collector.record_latency("total_request", -5.0)
        collector.record_latency("total_request", None)
        snap = collector.get_snapshot()
        assert snap["latency_ms"]["total_request"]["count"] == 0

    def test_multithreaded_increments_thread_safe(self):
        """Concurrent increments across multiple threads count accurately without race conditions."""
        collector = MetricsCollector()
        threads = 10
        increments_per_thread = 100

        def _worker():
            for _ in range(increments_per_thread):
                collector.increment("total_requests")
                collector.increment("query_requests")

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(_worker) for _ in range(threads)]
            for f in futures:
                f.result()

        snap = collector.get_snapshot()
        assert snap["requests"]["total"] == threads * increments_per_thread
        assert snap["requests"]["by_endpoint"]["/query"] == threads * increments_per_thread

    def test_singleton_reset(self):
        """get_metrics_collector returns singleton and reset_metrics clears counters."""
        reset_metrics()
        c1 = get_metrics_collector()
        c1.increment("total_requests", 50)
        assert c1.get_snapshot()["requests"]["total"] == 50

        reset_metrics()
        c2 = get_metrics_collector()
        assert c2.get_snapshot()["requests"]["total"] == 0
