"""Unit tests for GenerationService and LLM providers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.prompts import build_grounded_prompt, format_context_block
from app.generation.service import (
    GenerationError,
    GenerationService,
    MockLLMProvider,
    OpenAILLMProvider,
)
from app.retrieval.models import RetrievalResult


def _make_res(chunk_id: str, title: str, text: str) -> RetrievalResult:
    """Helper factory for RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="DOC001",
        text=text,
        title=title,
        department="Operations",
        similarity_score=0.9,
        rank=1,
    )


class TestGenerationService:
    """Unit test suite for GenerationService and LLM providers."""

    def test_mock_provider_generates_grounded_answer(self) -> None:
        """MockLLMProvider returns deterministic grounded answer from context."""
        provider = MockLLMProvider()
        service = GenerationService(provider=provider)

        context = ["Facility hours are 08:00 to 18:00 daily."]
        response = service.generate("What are facility hours?", context)

        assert isinstance(response, GenerationResponse)
        assert "Facility hours are 08:00 to 18:00 daily." in response.answer
        assert response.model_name == "mock-grounded-llm"

    def test_empty_query_raises_generation_error(self) -> None:
        """Empty query raises GenerationError."""
        service = GenerationService(provider=MockLLMProvider())

        with pytest.raises(GenerationError, match="empty"):
            service.generate("", ["Some context text."])

        with pytest.raises(GenerationError, match="empty"):
            service.generate("   ", ["Some context text."])

    def test_empty_context_returns_insufficient_info(self) -> None:
        """Empty context returns grounded insufficient information answer."""
        service = GenerationService(provider=MockLLMProvider())

        response = service.generate("What is the refund policy?", context=[])
        assert "sufficient information" in response.answer.lower()

        response2 = service.generate("What is the refund policy?", context="   ")
        assert "sufficient information" in response2.answer.lower()

    def test_provider_failure_wrapped_in_generation_error(self) -> None:
        """Exceptions raised by LLMProvider are wrapped in GenerationError."""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = RuntimeError("API connection timeout")
        mock_provider.model_name = "failing-model"

        service = GenerationService(provider=mock_provider)

        with pytest.raises(GenerationError, match="failed"):
            service.generate("Valid query", ["Valid context chunk."])

    def test_format_context_block(self) -> None:
        """format_context_block constructs clear source headers."""
        results = [
            _make_res("C1", "Title One", "Text for chunk one."),
            _make_res("C2", "Title Two", "Text for chunk two."),
        ]
        formatted = format_context_block(results)

        assert "[Source 1]" in formatted
        assert "Chunk ID: C1" in formatted
        assert "Title: Title One" in formatted
        assert "Text for chunk one." in formatted

        assert "[Source 2]" in formatted
        assert "Chunk ID: C2" in formatted
        assert "Text for chunk two." in formatted

    def test_build_grounded_prompt(self) -> None:
        """build_grounded_prompt includes system rules, context, and query."""
        prompt = build_grounded_prompt("What is X?", "Source context text.")

        assert "CRITICAL GROUNDEDNESS RULES" in prompt
        assert "Source context text." in prompt
        assert "What is X?" in prompt

    def test_openai_provider_missing_key_raises_error(self) -> None:
        """OpenAILLMProvider raises GenerationError if OPENAI_API_KEY is missing."""
        provider = OpenAILLMProvider(api_key="")

        with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
            provider.generate("prompt", "query", ["context"])
