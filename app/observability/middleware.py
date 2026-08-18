"""Phase 9 request correlation and structured observability middleware."""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.logging import log_structured_event
from app.observability.metrics import get_metrics_collector

logger = logging.getLogger("app.api.access")

_SAFE_REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _get_or_generate_request_id(request: Request) -> str:
    """Extract and validate X-Request-ID from headers, or generate a new UUID4."""
    header_val = request.headers.get("X-Request-ID")
    if header_val and _SAFE_REQUEST_ID_PATTERN.match(header_val.strip()):
        return header_val.strip()
    return str(uuid.uuid4())


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware providing request ID correlation, latency tracking, and metrics."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = _get_or_generate_request_id(request)
        request.state.request_id = request_id

        start_time = time.monotonic()
        metrics = get_metrics_collector()

        endpoint = request.url.path
        method = request.method

        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration_ms = round((time.monotonic() - start_time) * 1000.0, 2)

            # 1. Update Metrics
            metrics.increment("total_requests")
            if status_code < 400:
                metrics.increment("successful_requests")
            else:
                metrics.increment("failed_requests")

            # Endpoint-specific metric
            if endpoint == "/query":
                metrics.increment("query_requests")
            elif endpoint == "/retrieve":
                metrics.increment("retrieve_requests")
            elif endpoint == "/health":
                metrics.increment("health_requests")
            elif endpoint == "/ready":
                metrics.increment("ready_requests")
            elif endpoint == "/metrics":
                metrics.increment("metrics_requests")

            # Error code classification
            metrics.increment(f"status_{status_code}")
            if status_code == 422:
                metrics.increment("validation_errors")
            elif status_code == 404:
                metrics.increment("not_found_errors")
            elif status_code == 405:
                metrics.increment("method_not_allowed_errors")
            elif status_code >= 500:
                metrics.increment("internal_errors")

            metrics.record_latency("total_request", duration_ms)

            # 2. Structured JSON Access Log
            log_structured_event(
                logger=logger,
                event="http_request",
                level=logging.INFO if status_code < 400 else logging.WARNING,
                request_id=request_id,
                method=method,
                endpoint=endpoint,
                path=endpoint,
                status_code=status_code,
                duration_ms=duration_ms,
                latency_ms=duration_ms,
            )
