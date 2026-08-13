"""LLM Generation Service and Provider Abstraction.

Provides clean provider abstraction for LLM inference (MockLLMProvider for offline testing,
OpenAILLMProvider for live APIs) and GenerationService for grounded prompt execution.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from app.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)
from app.generation.models import GenerationResponse
from app.generation.prompts import build_grounded_prompt

logger = logging.getLogger(__name__)

INSUFFICIENT_INFO_ANSWER = (
    "I do not have sufficient information in the provided context to answer this question."
)


class GenerationError(Exception):
    """Exception raised for failures during LLM generation operations."""


class LLMProvider(ABC):
    """Abstract base class for LLM generation providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""

    @abstractmethod
    def generate(
        self, prompt: str, query: str, context_chunks: List[str]
    ) -> GenerationResponse:
        """Generate a response completion for the given prompt and query context.

        Parameters
        ----------
        prompt : str
            Full grounded system + context + query prompt text.
        query : str
            Raw user query text.
        context_chunks : List[str]
            Raw context text blocks.

        Returns
        -------
        GenerationResponse
            Structured generation response container.
        """


class MockLLMProvider(LLMProvider):
    """Deterministic offline LLM provider for testing and fallback execution."""

    def __init__(self, name: str = "mock-grounded-llm") -> None:
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    def generate(
        self, prompt: str, query: str, context_chunks: List[str]
    ) -> GenerationResponse:
        """Generate a deterministic grounded answer using context contents."""
        if not context_chunks or not any(c.strip() for c in context_chunks):
            return GenerationResponse(
                answer=INSUFFICIENT_INFO_ANSWER,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=len(INSUFFICIENT_INFO_ANSWER.split()),
            )

        # Build deterministic response from the first valid context chunk
        valid_chunks = [c for c in context_chunks if c.strip() and "No relevant context" not in c]
        if not valid_chunks:
            return GenerationResponse(
                answer=INSUFFICIENT_INFO_ANSWER,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=len(INSUFFICIENT_INFO_ANSWER.split()),
            )

        # Extract primary informational lines from context
        first_chunk = valid_chunks[0]
        lines = [line.strip() for line in first_chunk.splitlines() if line.strip() and not line.startswith("Document ID:") and not line.startswith("Chunk ID:") and not line.startswith("[Source")]

        content_snippet = " ".join(lines[:3]) if lines else first_chunk.strip()
        answer = f"Based on the provided documentation: {content_snippet}"

        return GenerationResponse(
            answer=answer,
            model_name=self.model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
        )


class OpenAILLMProvider(LLMProvider):
    """OpenAI API generation provider using standard urllib (no mandatory external SDK dependency)."""

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        api_key: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        api_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_url = api_url

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self, prompt: str, query: str, context_chunks: List[str]
    ) -> GenerationResponse:
        """Post chat completion request to OpenAI-compatible API endpoint."""
        if not self.api_key or not self.api_key.strip():
            raise GenerationError("OPENAI_API_KEY environment variable is missing or empty.")

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a grounded RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(self.api_url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_bytes = resp.read()
                result = json.loads(resp_bytes.decode("utf-8"))
        except Exception as exc:
            raise GenerationError(f"OpenAI API request failed: {exc}") from exc

        try:
            choices = result.get("choices", [])
            if not choices:
                raise GenerationError("Malformed OpenAI response: missing 'choices' list.")

            answer = choices[0]["message"]["content"].strip()
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            return GenerationResponse(
                answer=answer,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as exc:
            raise GenerationError(f"Failed to parse OpenAI API response payload: {exc}") from exc


class GenerationService:
    """Service handling grounded prompt construction and LLM answer generation."""

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        """Initialize GenerationService with configured or default provider.

        Parameters
        ----------
        provider : LLMProvider, optional
            Concrete LLM provider instance. Defaults to configured provider.
        """
        if provider is not None:
            self.provider = provider
        elif LLM_PROVIDER.lower() == "openai" and os.getenv("OPENAI_API_KEY"):
            self.provider = OpenAILLMProvider()
        else:
            self.provider = MockLLMProvider()

    def generate(
        self, query: str, context: Union[List[str], str]
    ) -> GenerationResponse:
        """Generate a grounded answer for query and context.

        Parameters
        ----------
        query : str
            User query string.
        context : Union[List[str], str]
            Context chunks list or formatted context block string.

        Returns
        -------
        GenerationResponse
            Structured generation response.

        Raises
        ------
        GenerationError
            If query is empty or generation provider fails.
        """
        if not query or not query.strip():
            raise GenerationError("Query string must not be empty.")

        clean_query = query.strip()

        # Parse context list
        if isinstance(context, str):
            formatted_context = context.strip()
            context_list = [formatted_context] if formatted_context else []
        else:
            context_list = [c for c in context if isinstance(c, str) and c.strip()]
            formatted_context = "\n\n".join(context_list)

        # Handle empty context gracefully
        if not context_list or not formatted_context or "No relevant context" in formatted_context:
            return GenerationResponse(
                answer=INSUFFICIENT_INFO_ANSWER,
                model_name=self.provider.model_name,
                prompt_tokens=0,
                completion_tokens=len(INSUFFICIENT_INFO_ANSWER.split()),
            )

        prompt = build_grounded_prompt(clean_query, formatted_context)

        try:
            return self.provider.generate(prompt, clean_query, context_list)
        except Exception as exc:
            if isinstance(exc, GenerationError):
                raise
            raise GenerationError(f"LLM provider generation failed: {exc}") from exc
