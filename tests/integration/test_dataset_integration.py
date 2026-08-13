"""
Integration test: load the entire Nexora dataset from disk and validate it.

This test requires the actual dataset files to be present under ``data/``.
It is marked with ``pytest.mark.integration`` and is skipped automatically
if the dataset files are not found (e.g. in CI without the dataset).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders import load_chunks_jsonl, load_documents_jsonl, load_questions_jsonl
from app.ingestion.validator import (
    EXPECTED_CHUNK_COUNT,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_QUESTION_DISTRIBUTION,
    validate_full_dataset,
)

# ---------------------------------------------------------------------------
# Resolve dataset paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data"
_DOCUMENTS_JSONL = _DATA_ROOT / "raw" / "documents.jsonl"
_CHUNKS_JSONL = _DATA_ROOT / "processed" / "chunks.jsonl"
_QUESTIONS_JSONL = _DATA_ROOT / "evaluation" / "questions.jsonl"
_DOCUMENTS_DIR = _DATA_ROOT / "raw" / "documents"

# ---------------------------------------------------------------------------
# Skip marker if files are absent
# ---------------------------------------------------------------------------

_DATASET_PRESENT = (
    _DOCUMENTS_JSONL.exists() and _CHUNKS_JSONL.exists() and _QUESTIONS_JSONL.exists()
)

pytestmark = pytest.mark.skipif(
    not _DATASET_PRESENT,
    reason="Nexora dataset not present under data/. Skipping integration tests.",
)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullDatasetIntegration:
    """Integration tests that load and validate the real Nexora dataset files."""

    def test_documents_count(self) -> None:
        """Exactly 30 documents must be loadable from documents.jsonl."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        assert len(documents) == EXPECTED_DOCUMENT_COUNT, (
            f"Expected {EXPECTED_DOCUMENT_COUNT} documents, got {len(documents)}."
        )

    def test_chunks_count(self) -> None:
        """Exactly 150 chunks must be loadable from chunks.jsonl."""
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        assert len(chunks) == EXPECTED_CHUNK_COUNT, (
            f"Expected {EXPECTED_CHUNK_COUNT} chunks, got {len(chunks)}."
        )

    def test_questions_count(self) -> None:
        """Exactly 150 questions must be loadable from questions.jsonl."""
        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        assert len(questions) == EXPECTED_QUESTION_COUNT, (
            f"Expected {EXPECTED_QUESTION_COUNT} questions, got {len(questions)}."
        )

    def test_document_ids_unique(self) -> None:
        """All document_id values in documents.jsonl must be unique."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        ids = [d.document_id for d in documents]
        assert len(ids) == len(set(ids)), "Duplicate document_id(s) found."

    def test_chunk_ids_unique(self) -> None:
        """All chunk_id values in chunks.jsonl must be unique."""
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_id(s) found."

    def test_question_ids_unique(self) -> None:
        """All question_id values in questions.jsonl must be unique."""
        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        ids = [q.question_id for q in questions]
        assert len(ids) == len(set(ids)), "Duplicate question_id(s) found."

    def test_question_type_distribution(self) -> None:
        """Question type counts must match the expected distribution exactly."""
        from collections import Counter

        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        dist = Counter(q.question_type for q in questions)

        for qtype, expected in EXPECTED_QUESTION_DISTRIBUTION.items():
            actual = dist.get(qtype, 0)
            assert actual == expected, (
                f"Question type '{qtype.value}': expected {expected}, got {actual}."
            )

    def test_chunk_document_references_valid(self) -> None:
        """Every chunk's document_id must reference a document in documents.jsonl."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        doc_ids = {d.document_id for d in documents}

        orphans = [c.chunk_id for c in chunks if c.document_id not in doc_ids]
        assert not orphans, (
            f"{len(orphans)} chunk(s) reference missing document IDs: {orphans[:5]}"
        )

    def test_question_document_references_valid(self) -> None:
        """Every question's relevant_documents must reference real documents."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        doc_ids = {d.document_id for d in documents}

        for q in questions:
            bad = [d for d in q.relevant_documents if d not in doc_ids]
            assert not bad, (
                f"Question '{q.question_id}' references missing document(s): {bad}"
            )

    def test_question_chunk_references_valid(self) -> None:
        """Every question's relevant_chunks must reference real chunks."""
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        chunk_ids = {c.chunk_id for c in chunks}

        for q in questions:
            bad = [c for c in q.relevant_chunks if c not in chunk_ids]
            assert not bad, (
                f"Question '{q.question_id}' references missing chunk(s): {bad}"
            )

    def test_individual_document_json_files(self) -> None:
        """Every DOC*.json file must be loadable and validate correctly."""
        from app.ingestion.loaders import load_document_json

        json_files = sorted(_DOCUMENTS_DIR.glob("DOC*.json"))
        assert len(json_files) == EXPECTED_DOCUMENT_COUNT, (
            f"Expected {EXPECTED_DOCUMENT_COUNT} individual JSON files, "
            f"found {len(json_files)}."
        )
        for path in json_files:
            doc = load_document_json(path)
            assert doc.document_id, f"document_id missing in {path.name}"
            assert doc.text.strip(), f"Empty text in {path.name}"

    def test_full_dataset_validation_passes(self) -> None:
        """The :func:`validate_full_dataset` function must pass for the real dataset."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))

        # Must not raise DatasetValidationError
        validate_full_dataset(documents, chunks, questions)

    def test_all_documents_have_non_empty_fields(self) -> None:
        """Every document must have non-empty required string fields."""
        documents = list(load_documents_jsonl(_DOCUMENTS_JSONL))
        for doc in documents:
            assert doc.title.strip(), f"Empty title in {doc.document_id}"
            assert doc.text.strip(), f"Empty text in {doc.document_id}"
            assert doc.source.strip(), f"Empty source in {doc.document_id}"
            assert doc.department.strip(), f"Empty department in {doc.document_id}"
            assert doc.owner.strip(), f"Empty owner in {doc.document_id}"
            assert doc.module_code.strip(), f"Empty module_code in {doc.document_id}"

    def test_all_chunks_have_non_empty_text(self) -> None:
        """Every chunk must have non-empty text."""
        chunks = list(load_chunks_jsonl(_CHUNKS_JSONL))
        for chunk in chunks:
            assert chunk.text.strip(), f"Empty text in {chunk.chunk_id}"

    def test_unanswerable_questions_have_empty_chunks(self) -> None:
        """All unanswerable questions must have empty relevant_chunks."""
        from app.ingestion.models import QuestionType

        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        for q in questions:
            if q.question_type == QuestionType.UNANSWERABLE:
                assert q.relevant_chunks == [], (
                    f"Unanswerable question '{q.question_id}' has non-empty relevant_chunks."
                )

    def test_answerable_questions_have_chunks(self) -> None:
        """All answerable (non-unanswerable) questions must have at least one chunk."""
        from app.ingestion.models import QuestionType

        questions = list(load_questions_jsonl(_QUESTIONS_JSONL))
        for q in questions:
            if q.question_type != QuestionType.UNANSWERABLE:
                assert q.relevant_chunks, (
                    f"Answerable question '{q.question_id}' has no relevant_chunks."
                )
