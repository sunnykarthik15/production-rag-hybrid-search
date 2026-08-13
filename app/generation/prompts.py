"""Prompt design and context formatting for grounded LLM generation."""

from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.models import RetrievalResult

SYSTEM_PROMPT = (
    "You are a strict, factual RAG assistant for Nexora documentation.\n"
    "Your objective is to answer the user query accurately using ONLY the provided context sources.\n\n"
    "CRITICAL GROUNDEDNESS RULES:\n"
    "1. Base your answer strictly on the provided Context Sources below.\n"
    "2. Do NOT invent, assume, or extrapolate facts not explicitly stated in the context.\n"
    "3. If the context does NOT contain sufficient evidence to answer the question, state explicitly:\n"
    "   'I do not have sufficient information in the provided context to answer this question.'\n"
    "4. Keep your answer factual, precise, and concise.\n"
    "5. Do not include metadata headers (e.g., 'Document ID', 'Chunk ID') in your final answer body."
)


def format_context_block(results: List[RetrievalResult]) -> str:
    """Format retrieved candidate chunks into a structured context string for prompt injection.

    Parameters
    ----------
    results : List[RetrievalResult]
        List of reranked candidate retrieval result objects.

    Returns
    -------
    str
        Formatted context block string detailing source metadata and chunk text content.
    """
    if not results:
        return "No relevant context sources available."

    blocks: List[str] = []
    for rank, item in enumerate(results, start=1):
        block = (
            f"[Source {rank}]\n"
            f"Document ID: {item.document_id}\n"
            f"Chunk ID: {item.chunk_id}\n"
            f"Title: {item.title}\n"
            f"Department: {item.department}\n"
            f"Content:\n{item.text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_grounded_prompt(query: str, formatted_context: str) -> str:
    """Combine system guidelines, formatted context blocks, and user query into a final prompt string.

    Parameters
    ----------
    query : str
        User query text.
    formatted_context : str
        Formatted context text block.

    Returns
    -------
    str
        Full prompt ready for LLM consumption.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CONTEXT SOURCES ===\n"
        f"{formatted_context}\n\n"
        f"=== USER QUERY ===\n"
        f"{query}\n\n"
        f"=== ANSWER ==="
    )
