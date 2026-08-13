"""
Shared pytest fixtures for the Nexora RAG test suite.

Fixtures defined here are available to both unit and integration tests
without explicit imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from app.ingestion.models import Chunk, Document, EvaluationQuestion, QuestionType


# ---------------------------------------------------------------------------
# Repository root helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return _REPO_ROOT


@pytest.fixture(scope="session")
def data_root(repo_root: Path) -> Path:
    """Return the absolute path to the ``data/`` directory."""
    return repo_root / "data"


# ---------------------------------------------------------------------------
# Minimal in-memory fixtures for unit tests
# ---------------------------------------------------------------------------


def _make_document(**kwargs) -> Document:
    """Build a valid :class:`Document` with sensible defaults."""
    defaults = {
        "document_id": "DOC001",
        "title": "Test Document",
        "department": "Testing",
        "topic": "unit tests",
        "owner": "Test Team",
        "module_code": "NX-TST-001",
        "text": "This is the document body text.",
        "source": "Nexora Synthetic Operations Corpus",
        "version": "1.0",
    }
    defaults.update(kwargs)
    return Document(**defaults)


def _make_chunk(**kwargs) -> Chunk:
    """Build a valid :class:`Chunk` with sensible defaults."""
    defaults = {
        "chunk_id": "DOC001_CHUNK_01",
        "document_id": "DOC001",
        "title": "Test Document",
        "department": "Testing",
        "text": "Chunk body text.",
        "chunk_index": 1,
        "source": "Nexora Synthetic Operations Corpus",
    }
    defaults.update(kwargs)
    return Chunk(**defaults)


def _make_question(**kwargs) -> EvaluationQuestion:
    """Build a valid :class:`EvaluationQuestion` with sensible defaults."""
    defaults = {
        "question_id": "Q001",
        "question": "What is the module code?",
        "question_type": QuestionType.DIRECT,
        "expected_answer": "NX-TST-001",
        "relevant_documents": ["DOC001"],
        "relevant_chunks": ["DOC001_CHUNK_01"],
        "answerable": True,
    }
    defaults.update(kwargs)
    return EvaluationQuestion(**defaults)


@pytest.fixture
def valid_document() -> Document:
    """A single valid Document."""
    return _make_document()


@pytest.fixture
def valid_chunk() -> Chunk:
    """A single valid Chunk."""
    return _make_chunk()


@pytest.fixture
def valid_question() -> EvaluationQuestion:
    """A single valid answerable EvaluationQuestion."""
    return _make_question()


@pytest.fixture
def thirty_documents() -> List[Document]:
    """Exactly 30 valid, unique Documents."""
    return [
        _make_document(
            document_id=f"DOC{i:03d}",
            title=f"Document {i}",
            module_code=f"NX-TST-{i:03d}",
        )
        for i in range(1, 31)
    ]


@pytest.fixture
def one_fifty_chunks(thirty_documents: List[Document]) -> List[Chunk]:
    """Exactly 150 valid Chunks (5 per document)."""
    chunks: List[Chunk] = []
    for doc in thirty_documents:
        for idx in range(1, 6):
            chunks.append(
                _make_chunk(
                    chunk_id=f"{doc.document_id}_CHUNK_{idx:02d}",
                    document_id=doc.document_id,
                    title=doc.title,
                    chunk_index=idx,
                )
            )
    return chunks


@pytest.fixture
def one_fifty_questions(
    thirty_documents: List[Document],
    one_fifty_chunks: List[Chunk],
) -> List[EvaluationQuestion]:
    """Exactly 150 valid questions with the correct type distribution."""
    # Actual distribution: 25 per type × 6 types = 150 total
    type_sequence = (
        [QuestionType.DIRECT] * 25
        + [QuestionType.SEMANTIC] * 25
        + [QuestionType.EXACT_KEYWORD] * 25
        + [QuestionType.MULTI_DOCUMENT] * 25
        + [QuestionType.DISTRACTOR] * 25
        + [QuestionType.UNANSWERABLE] * 25
    )

    questions: List[EvaluationQuestion] = []
    for i, qtype in enumerate(type_sequence, start=1):
        doc = thirty_documents[(i - 1) % 30]
        chunk = one_fifty_chunks[(i - 1) % 150]

        if qtype == QuestionType.UNANSWERABLE:
            questions.append(
                _make_question(
                    question_id=f"Q{i:03d}",
                    question_type=qtype,
                    expected_answer="The corpus does not contain this information.",
                    relevant_documents=[doc.document_id],
                    relevant_chunks=[],
                    answerable=False,
                )
            )
        else:
            questions.append(
                _make_question(
                    question_id=f"Q{i:03d}",
                    question_type=qtype,
                    relevant_documents=[doc.document_id],
                    relevant_chunks=[chunk.chunk_id],
                    answerable=True,
                )
            )
    return questions


# ---------------------------------------------------------------------------
# Temp-file fixtures for loader tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_documents_jsonl(tmp_path: Path, thirty_documents: List[Document]) -> Path:
    """Write 30 valid documents to a temporary JSONL file."""
    path = tmp_path / "documents.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for doc in thirty_documents:
            fh.write(doc.model_dump_json() + "\n")
    return path


@pytest.fixture
def tmp_chunks_jsonl(tmp_path: Path, one_fifty_chunks: List[Chunk]) -> Path:
    """Write 150 valid chunks to a temporary JSONL file."""
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for chunk in one_fifty_chunks:
            fh.write(chunk.model_dump_json() + "\n")
    return path


@pytest.fixture
def tmp_questions_jsonl(
    tmp_path: Path, one_fifty_questions: List[EvaluationQuestion]
) -> Path:
    """Write 150 valid questions to a temporary JSONL file."""
    path = tmp_path / "questions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for q in one_fifty_questions:
            fh.write(q.model_dump_json() + "\n")
    return path


@pytest.fixture
def tmp_malformed_jsonl(tmp_path: Path) -> Path:
    """A JSONL file where one line is malformed JSON."""
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"document_id": "DOC001"}\nNOT_VALID_JSON\n', encoding="utf-8")
    return path
