"""Ingestion module — public API surface for Phase 1.

Exports the data models and loaders that are ready for use.
"""

from app.ingestion.loaders import (
    DataLoadError,
    load_chunks_jsonl,
    load_document_json,
    load_documents_jsonl,
    load_questions_jsonl,
)
from app.ingestion.models import Chunk, Document, EvaluationQuestion, QuestionType

__all__ = [
    # Models
    "Document",
    "Chunk",
    "EvaluationQuestion",
    "QuestionType",
    # Loaders
    "load_documents_jsonl",
    "load_document_json",
    "load_chunks_jsonl",
    "load_questions_jsonl",
    # Exceptions
    "DataLoadError",
]
