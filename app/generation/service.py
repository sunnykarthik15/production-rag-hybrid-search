"""LLM Generation Service and Provider Abstraction.

Provides clean provider abstraction for LLM inference (MockLLMProvider for offline testing,
OpenAILLMProvider for live APIs) and GenerationService for grounded prompt execution.
"""

from __future__ import annotations

import json
import logging
import os
import re
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


def _apply_paraphrase(text: str) -> str:
    """Deterministically paraphrase text using synonym mappings while preserving facts."""
    paraphrase_map = [
        (r"\boperating hours are\b", "operational schedule is"),
        (r"\boperating hours\b", "service hours"),
        (r"\bannual\b", "yearly"),
        (r"\bfee is\b", "charge is"),
        (r"\bpermits cost\b", "passes are priced at"),
        (r"\bmust be submitted before\b", "should be submitted prior to"),
        (r"\baccess is restricted to\b", "entry is limited exclusively to"),
        (r"\bmaintenance is scheduled for\b", "routine maintenance is planned for"),
        (r"\bprovides 24/7 online\b", "offers continuous round-the-clock online"),
        (r"\bcloses at\b", "concludes daily operations at"),
        (r"\bunder module\b", "governed by specification"),
        (r"\ballow(?:ed|s)?\b", "permit"),
    ]
    res = text
    for pat, repl in paraphrase_map:
        res = re.sub(pat, repl, res, flags=re.IGNORECASE)
    return res


def _apply_numeric_mutation(text: str) -> str:
    """Deterministically mutate numeric quantities, currency, and percentages."""
    mutations = [
        (r"\$50(?:\.00)?\b", "$80.00"),
        (r"\$100\b", "$250"),
        (r"\b24/7\b", "12/5"),
        (r"\b100%\b", "75%"),
        (r"\b50%\b", "25%"),
        (r"\b10\b", "25"),
        (r"\b30\b", "60"),
        (r"\b4\s*hours\b", "8 hours"),
        (r"\b3\.5\b", "7.5"),
    ]
    res = text
    for pat, repl in mutations:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    # Generic fallback: if digits exist, increment first digit
    if re.search(r"\b\d+\b", res):
        return re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 15), res, count=1)
    return res + " with an extra fee of $85"


def _apply_date_mutation(text: str) -> str:
    """Deterministically mutate calendar dates."""
    mutations = [
        (r"\bDec(?:ember)?\s+12\b", "Dec 15"),
        (r"\bJan(?:uary)?\s+15\b", "Jan 28"),
        (r"\bFeb(?:ruary)?\s+1\b", "Feb 18"),
        (r"\bMar(?:ch)?\s+15\b", "Mar 29"),
        (r"\b2024\b", "2029"),
        (r"\b2023\b", "2028"),
    ]
    res = text
    for pat, repl in mutations:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return res + " scheduled on November 15, 2029"


def _apply_weekday_mutation(text: str) -> str:
    """Deterministically mutate days of week."""
    mutations = [
        (r"\bFriday\b", "Wednesday"),
        (r"\bMonday\b", "Thursday"),
        (r"\bTuesday\b", "Saturday"),
        (r"\bWednesday\b", "Monday"),
        (r"\bThursday\b", "Sunday"),
        (r"\bSaturday\b", "Tuesday"),
        (r"\bSunday\b", "Friday"),
    ]
    res = text
    for pat, repl in mutations:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return res + " on Wednesday"


def _apply_time_mutation(text: str) -> str:
    """Deterministically mutate clock times."""
    mutations = [
        (r"\b5\s*PM\b", "10 PM"),
        (r"\b9\s*AM\b", "11 AM"),
        (r"\b17:00\b", "21:00"),
        (r"\b08:00\b", "11:30"),
        (r"\b10:00\b", "14:00"),
    ]
    res = text
    for pat, repl in mutations:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return res + " at 11:30 PM"


def _apply_temporal_mutation(text: str) -> str:
    """Deterministically mutate any temporal expression (weekday, date, time)."""
    res = _apply_weekday_mutation(text)
    if res != text:
        return res
    res = _apply_date_mutation(text)
    if res != text:
        return res
    return _apply_time_mutation(text)


def _apply_polarity_inversion(text: str) -> str:
    """Deterministically invert polarity, permission, and restriction terms."""
    inversions = [
        (r"\brestricted\b", "unrestricted"),
        (r"\bunrestricted\b", "restricted"),
        (r"\bpermitted\b", "prohibited"),
        (r"\bprohibited\b", "permitted"),
        (r"\ballowed\b", "forbidden"),
        (r"\bforbidden\b", "allowed"),
        (r"\bclosed\b", "open"),
        (r"\bopen\b", "closed"),
        (r"\bmandatory\b", "optional"),
        (r"\boptional\b", "mandatory"),
        (r"\brequired\b", "exempt"),
        (r"\bexempt\b", "required"),
        (r"\bauthorized\b", "unauthorized"),
        (r"\bunauthorized\b", "authorized"),
    ]
    res = text
    for pat, repl in inversions:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return "It is prohibited: " + res


def _apply_negation_insertion(text: str) -> str:
    """Deterministically insert negation into affirmative assertions."""
    replacements = [
        (r"\brequires\b", "does not require"),
        (r"\brequire\b", "do not require"),
        (r"\bis required\b", "is not required"),
        (r"\bare required\b", "are not required"),
        (r"\bis permitted\b", "is not permitted"),
        (r"\bare permitted\b", "are not permitted"),
        (r"\bis allowed\b", "is not allowed"),
        (r"\bare allowed\b", "are not allowed"),
        (r"\bis restricted\b", "is not restricted"),
        (r"\bprovides\b", "does not provide"),
        (r"\boperates\b", "does not operate"),
    ]
    res = text
    for pat, repl in replacements:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return "It is not the case that " + text


def _apply_negation_removal(text: str) -> str:
    """Deterministically remove negation from negative assertions."""
    replacements = [
        (r"\bdoes not require\b", "requires"),
        (r"\bdo not require\b", "require"),
        (r"\bis not required\b", "is required"),
        (r"\bare not required\b", "are required"),
        (r"\bis not permitted\b", "is permitted"),
        (r"\bare not permitted\b", "are permitted"),
        (r"\bis not allowed\b", "is allowed"),
        (r"\bare not allowed\b", "are allowed"),
        (r"\bis restricted\b", "is unrestricted"),
        (r"\brestricted\b", "unrestricted"),
        (r"\bprohibited\b", "permitted"),
        (r"\bforbidden\b", "permitted"),
        (r"\bclosed\b", "open"),
        (r"\bwithout\b", "with"),
    ]
    res = text
    for pat, repl in replacements:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bnot\s+", "", text, flags=re.IGNORECASE)
    if cleaned != text:
        return cleaned
    return "It is permitted: " + text


def _apply_entity_substitution(text: str) -> str:
    """Deterministically substitute module codes and identifiers."""
    substitutions = [
        (r"\bNX-FAC-100\b", "NX-SEC-999"),
        (r"\bNX-LIB-101\b", "NX-ENG-404"),
        (r"\bNX-SEC-201\b", "NX-OPS-000"),
        (r"\bNX-[A-Z]{3}-\d+\b", "NX-MUT-999"),
    ]
    res = text
    for pat, repl in substitutions:
        if re.search(pat, res, flags=re.IGNORECASE):
            return re.sub(pat, repl, res, count=1, flags=re.IGNORECASE)
    return res + " under protocol NX-MUT-999"


class MockLLMProvider(LLMProvider):
    """Deterministic offline LLM provider for testing, benchmark evaluation, and simulation."""

    def __init__(
        self,
        name: str = "mock-grounded-llm",
        include_citations: bool = False,
        mode: str = "direct_copy",
    ) -> None:
        self._name = name
        self.include_citations = include_citations
        self.mode = mode

    @property
    def model_name(self) -> str:
        return self._name

    def generate(
        self, prompt: str, query: str, context_chunks: List[str]
    ) -> GenerationResponse:
        """Generate a deterministic answer using context contents and configured simulation mode."""
        if not context_chunks or not any(c.strip() for c in context_chunks):
            return GenerationResponse(
                answer=INSUFFICIENT_INFO_ANSWER,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=len(INSUFFICIENT_INFO_ANSWER.split()),
            )

        # Filter valid candidate chunks
        valid_chunks = [c for c in context_chunks if c.strip() and "No relevant context" not in c]
        if not valid_chunks:
            return GenerationResponse(
                answer=INSUFFICIENT_INFO_ANSWER,
                model_name=self.model_name,
                prompt_tokens=0,
                completion_tokens=len(INSUFFICIENT_INFO_ANSWER.split()),
            )

        def _extract_chunk_lines(chunk_str: str) -> List[str]:
            if "Content:\n" in chunk_str:
                content_part = chunk_str.split("Content:\n", 1)[1]
                if "\n[Source " in content_part:
                    content_part = content_part.split("\n[Source ", 1)[0]
                raw_lines = [line.strip() for line in content_part.splitlines() if line.strip()]
            elif "\nContent:" in chunk_str:
                content_part = chunk_str.split("\nContent:", 1)[1]
                if "\n[Source " in content_part:
                    content_part = content_part.split("\n[Source ", 1)[0]
                raw_lines = [line.strip() for line in content_part.splitlines() if line.strip()]
            else:
                raw_lines = [line.strip() for line in chunk_str.splitlines() if line.strip()]

            return [
                line.strip()
                for line in raw_lines
                if line.strip()
                and not line.startswith("Document ID:")
                and not line.startswith("Chunk ID:")
                and not line.startswith("Title:")
                and not line.startswith("Department:")
                and not line.startswith("[Source")
            ]

        def _extract_rank(chunk_str: str, default_rank: int = 1) -> int:
            if "[Source " in chunk_str:
                try:
                    return int(chunk_str.split("[Source ")[1].split("]")[0])
                except (IndexError, ValueError):
                    return default_rank
            return default_rank

        # Extract primary informational lines from first valid chunk
        first_chunk = valid_chunks[0]
        lines = _extract_chunk_lines(first_chunk)
        source_rank = _extract_rank(first_chunk, default_rank=1)

        raw_content = " ".join(lines[:3]) if lines else first_chunk.strip()
        raw_sentences = [
            s.strip().rstrip(".")
            for s in re.split(r"(?<=[.!?])\s+", raw_content)
            if s.strip()
        ]
        if not raw_sentences:
            raw_sentences = [raw_content.rstrip(".")]

        # Determine effective citation mode
        attach_citations = self.include_citations or ("CITATION RULES" in prompt and self.mode != "missing_citations")

        # Apply simulation transformations based on configured mode
        mode_lower = self.mode.lower()

        if mode_lower in {"faithful_paraphrase", "paraphrase"}:
            processed_sentences = [_apply_paraphrase(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "missing_citations":
            processed_sentences = [_apply_paraphrase(s) for s in raw_sentences]
            attach_citations = False
            cited_rank = source_rank
        elif mode_lower in {"wrong_citation", "wrong_citations"}:
            processed_sentences = [_apply_paraphrase(s) for s in raw_sentences]
            # Deliberately cite incorrect source rank (e.g. source 2 instead of 1)
            cited_rank = (source_rank % max(len(valid_chunks), 2)) + 1
            if cited_rank == source_rank:
                cited_rank = source_rank + 1
        elif mode_lower in {"invalid_citation", "invalid_citations"}:
            processed_sentences = [_apply_paraphrase(s) for s in raw_sentences]
            cited_rank = 99  # Out of range
        elif mode_lower in {"fabricated_number", "numeric_mutation"}:
            processed_sentences = [_apply_numeric_mutation(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "fabricated_date":
            processed_sentences = [_apply_date_mutation(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "fabricated_weekday":
            processed_sentences = [_apply_weekday_mutation(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "fabricated_time":
            processed_sentences = [_apply_time_mutation(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "temporal_mutation":
            processed_sentences = [_apply_temporal_mutation(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "polarity_inversion":
            processed_sentences = [_apply_polarity_inversion(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "negation_insertion":
            processed_sentences = [_apply_negation_insertion(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "negation_removal":
            processed_sentences = [_apply_negation_removal(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower in {"unsupported_entity", "entity_substitution"}:
            processed_sentences = [_apply_entity_substitution(s) for s in raw_sentences]
            cited_rank = source_rank
        elif mode_lower == "unsupported_claim":
            processed_sentences = list(raw_sentences) + [
                "Quantum propulsion units require mandatory sub-orbital shielding"
            ]
            cited_rank = source_rank
        elif mode_lower == "multi_source_misattribution":
            # Multi-source answer where citations are swapped across chunks
            if len(valid_chunks) >= 2:
                lines2 = _extract_chunk_lines(valid_chunks[1])
                s1 = raw_sentences[0] if raw_sentences else "Operation standard is active"
                s2 = lines2[0] if lines2 else "Secondary protocol applies"
                # Misattribute chunk 1 sentence to Source 2, and chunk 2 sentence to Source 1
                answer_body = f"{_apply_paraphrase(s1)} [Source 2]. {_apply_paraphrase(s2)} [Source 1]."
            else:
                s1 = raw_sentences[0] if raw_sentences else "Operation standard is active"
                answer_body = f"{_apply_paraphrase(s1)} [Source 2]."
            answer = f"Based on the provided documentation: {answer_body}"
            return GenerationResponse(
                answer=answer,
                model_name=self.model_name,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(answer.split()),
            )
        else:
            # direct_copy legacy mode
            processed_sentences = raw_sentences
            cited_rank = source_rank

        if attach_citations:
            cited_sentences = [f"{s} [Source {cited_rank}]." for s in processed_sentences]
            answer_body = " ".join(cited_sentences)
            answer = f"Based on the provided documentation: {answer_body}"
        else:
            answer_body = ". ".join(processed_sentences) + "."
            answer = f"Based on the provided documentation: {answer_body}"

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
