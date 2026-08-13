"""LLM Generation package for grounded prompt construction and answer generation."""

from app.generation.models import GenerationRequest, GenerationResponse
from app.generation.prompts import build_grounded_prompt, format_context_block
from app.generation.service import (
    GenerationError,
    GenerationService,
    LLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
)

__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "GenerationService",
    "GenerationError",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAILLMProvider",
    "build_grounded_prompt",
    "format_context_block",
]
