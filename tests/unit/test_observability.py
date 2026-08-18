"""Unit tests for Phase 9 structured JSON logging and privacy-safe data sanitization."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from app.observability.logging import (
    StructuredJSONFormatter,
    get_query_hash,
    log_structured_event,
    sanitize_log_data,
)


class TestQueryHashing:
    """Test deterministic SHA-256 query hashing."""

    def test_query_hash_deterministic(self):
        """Identical queries produce the exact same SHA-256 hash."""
        h1 = get_query_hash("What is the campus operating policy?")
        h2 = get_query_hash("What is the campus operating policy?")
        assert h1 == h2
        assert len(h1) == 64

    def test_query_hash_whitespace_normalized(self):
        """Leading and trailing whitespace are normalized before hashing."""
        h1 = get_query_hash("What is the campus operating policy?")
        h2 = get_query_hash("   What is the campus operating policy?   ")
        assert h1 == h2

    def test_empty_query_hash(self):
        """Empty query returns empty string."""
        assert get_query_hash("") == ""


class TestLogSanitization:
    """Test sensitive data, token, and filesystem path redaction."""

    def test_redact_openai_key(self):
        """OpenAI API keys matching sk-... are redacted."""
        raw = "Error authenticating with sk-proj1234567890abcdef1234567890"
        sanitized = sanitize_log_data(raw)
        assert "sk-proj" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_redact_bearer_token(self):
        """Bearer auth headers are redacted."""
        raw = "Authorization: Bearer my_secret_token_123"
        sanitized = sanitize_log_data(raw)
        assert "my_secret_token" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_redact_windows_paths(self):
        """Windows drive letter paths with / and \\ are redacted."""
        raw_bs = "File not found at F:\\RAG-Project 1\\data\\vector_store\\index.faiss"
        raw_fs = "File not found at C:/Users/Admin/RAG-Project/data.json"
        assert "[PATH_REDACTED]" in sanitize_log_data(raw_bs)
        assert "[PATH_REDACTED]" in sanitize_log_data(raw_fs)

    def test_redact_posix_paths(self):
        """POSIX root paths are redacted."""
        raw = "Error reading /home/app/config.json"
        assert "[PATH_REDACTED]" in sanitize_log_data(raw)

    def test_recursive_dict_sanitization(self):
        """Nested dictionaries and lists are recursively sanitized."""
        data = {
            "token": "sk-12345678901234567890",
            "details": {
                "path": "F:\\secret\\file.txt",
                "nested_list": ["Bearer token123", "normal text"],
            },
        }
        sanitized = sanitize_log_data(data)
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["details"]["path"] == "[PATH_REDACTED]"
        assert sanitized["details"]["nested_list"][0] == "[REDACTED]"
        assert sanitized["details"]["nested_list"][1] == "normal text"


class TestStructuredJSONFormatter:
    """Test StructuredJSONFormatter output structure."""

    def test_format_log_record(self):
        """Formatted record is valid JSON containing timestamp, level, logger, and message."""
        formatter = StructuredJSONFormatter()
        logger = logging.getLogger("test_logger")
        record = logger.makeRecord(
            name="test_logger",
            level=logging.INFO,
            fn="test.py",
            lno=10,
            msg="User search executed",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"request_id": "REQ-123", "latency_ms": 42.5}
        formatted = formatter.format(record)

        data = json.loads(formatted)
        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "User search executed"
        assert data["request_id"] == "REQ-123"
        assert data["latency_ms"] == 42.5
        assert "timestamp" in data

    def test_log_structured_event_respects_level(self):
        """log_structured_event does not emit if logger is disabled for that level."""
        mock_logger = MagicMock()
        mock_logger.isEnabledFor.return_value = False

        log_structured_event(mock_logger, "test_event", level=logging.DEBUG)
        mock_logger.handle.assert_not_called()
