"""Structured JSON logging and privacy-preserving request logging for Phase 9."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
from typing import Any, Dict, Optional

# Regex patterns for secret and path redaction in logs
_SECRET_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(OPENAI_API_KEY\s*=\s*\S+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+\S+)", re.IGNORECASE),
    re.compile(r"([a-f0-9]{32,})", re.IGNORECASE),
]

_PATH_PATTERN = re.compile(
    r"[A-Z]:[/\\][^\s\"']+|/(?:home|usr|var|tmp|etc|opt|app)[/\\]?[^\s\"']*",
    re.IGNORECASE,
)


def get_query_hash(query: str) -> str:
    """Compute deterministic SHA-256 hash of a query string for privacy-safe tracking.

    Parameters
    ----------
    query : str
        User query text.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.
    """
    if not query:
        return ""
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


def sanitize_log_data(value: Any) -> Any:
    """Recursively sanitize strings, dictionaries, and lists to strip secrets and file paths."""
    if isinstance(value, str):
        sanitized = value
        for pattern in _SECRET_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        sanitized = _PATH_PATTERN.sub("[PATH_REDACTED]", sanitized)
        return sanitized
    elif isinstance(value, dict):
        return {k: sanitize_log_data(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [sanitize_log_data(item) for item in value]
    return value


class StructuredJSONFormatter(logging.Formatter):
    """Custom log formatter that outputs log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include custom extra fields attached to the log record
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            for k, v in record.structured_data.items():
                log_entry[k] = sanitize_log_data(v)

        return json.dumps(log_entry, default=str)


def log_structured_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """Emit a structured JSON log entry.

    Parameters
    ----------
    logger : logging.Logger
        Target logger instance.
    event : str
        Human-readable event name.
    level : int, optional
        Logging level (default: logging.INFO).
    **kwargs : Any
        Key-value attributes attached to the structured log entry.
    """
    if not logger.isEnabledFor(level):
        return

    sanitized_kwargs = sanitize_log_data(kwargs)
    record = logger.makeRecord(
        name=logger.name,
        level=level,
        fn="",
        lno=0,
        msg=event,
        args=(),
        exc_info=None,
        extra={"structured_data": {"event": event, **sanitized_kwargs}},
    )
    logger.handle(record)
