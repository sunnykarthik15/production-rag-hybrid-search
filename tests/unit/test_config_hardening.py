"""Unit tests for Phase 9 configuration parsing hardening and safe fallbacks."""

from __future__ import annotations

import os
from unittest.mock import patch

from app.config import _get_env_int, _get_env_list


class TestConfigHardening:
    """Test safe parsing of environment variables with bounds and fallback defaults."""

    def test_get_env_int_valid(self):
        """Valid integer strings within bounds are parsed correctly."""
        with patch.dict(os.environ, {"TEST_PORT": "8080"}):
            assert _get_env_int("TEST_PORT", default=8000, min_val=1, max_val=65535) == 8080

    def test_get_env_int_missing_uses_default(self):
        """Missing environment variable returns the specified default."""
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env_int("NONEXISTENT_KEY", default=9000) == 9000

    def test_get_env_int_invalid_string_uses_default(self):
        """Non-numeric string safely returns default without raising ValueError."""
        with patch.dict(os.environ, {"TEST_PORT": "invalid_number"}):
            assert _get_env_int("TEST_PORT", default=8000) == 8000

    def test_get_env_int_below_minimum_uses_default(self):
        """Value below minimum boundary returns default."""
        with patch.dict(os.environ, {"TEST_PORT": "0"}):
            assert _get_env_int("TEST_PORT", default=8000, min_val=1, max_val=65535) == 8000

    def test_get_env_int_above_maximum_uses_default(self):
        """Value above maximum boundary returns default."""
        with patch.dict(os.environ, {"TEST_PORT": "70000"}):
            assert _get_env_int("TEST_PORT", default=8000, min_val=1, max_val=65535) == 8000

    def test_get_env_list_valid_comma_string(self):
        """Comma-separated string is parsed and stripped into a list of strings."""
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://example.com, http://localhost:3000 "}):
            parsed = _get_env_list("ALLOWED_ORIGINS", default=["http://localhost:8000"])
            assert parsed == ["https://example.com", "http://localhost:3000"]

    def test_get_env_list_empty_uses_default(self):
        """Empty or whitespace-only environment variable returns default list."""
        with patch.dict(os.environ, {"ALLOWED_ORIGINS": "   "}):
            parsed = _get_env_list("ALLOWED_ORIGINS", default=["http://localhost:8000"])
            assert parsed == ["http://localhost:8000"]
