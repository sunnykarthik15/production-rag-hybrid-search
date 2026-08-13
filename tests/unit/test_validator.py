"""Unit tests for the dataset validator (validator.py)."""

from __future__ import annotations

from typing import List

import pytest

from app.ingestion.models import Chunk, Document, EvaluationQuestion, QuestionType
from app.ingestion.validator import (
    EXPECTED_CHUNK_COUNT,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUESTION_COUNT,
    DatasetValidationError,
    validate_chunks,
    validate_cross_references,
    validate_documents,
    validate_full_dataset,
    validate_questions,
)


# ---------------------------------------------------------------------------
# validate_documents
# ---------------------------------------------------------------------------


class TestValidateDocuments:
    """Tests for :func:`validate_documents`."""

    def test_valid_set_passes(self, thirty_documents: List[Document]) -> None:
        """Exactly 30 valid, unique documents must pass."""
        validate_documents(thirty_documents)  # must not raise

    def test_wrong_count_raises(self, thirty_documents: List[Document]) -> None:
        """Fewer than 30 documents must raise DatasetValidationError."""
        with pytest.raises(DatasetValidationError, match=str(EXPECTED_DOCUMENT_COUNT)):
            validate_documents(thirty_documents[:29])

    def test_too_many_raises(self, thirty_documents: List[Document], valid_document: Document) -> None:
        """More than 30 documents must raise DatasetValidationError."""
        extra = [*thirty_documents, valid_document]
        with pytest.raises(DatasetValidationError):
            validate_documents(extra)

    def test_duplicate_ids_raises(self, thirty_documents: List[Document]) -> None:
        """Duplicate document_id values must be flagged."""
        duplicate = thirty_documents[0]  # same ID as first entry
        with pytest.raises(DatasetValidationError, match="Duplicate document_id"):
            validate_documents([*thirty_documents, duplicate])

    def test_empty_collection_raises(self) -> None:
        """An empty list must raise DatasetValidationError."""
        with pytest.raises(DatasetValidationError):
            validate_documents([])

    def test_failures_collected_together(self, thirty_documents: List[Document]) -> None:
        """When both count and duplicate errors exist, both are in failures."""
        duplicate = thirty_documents[0]
        bad_set = [*thirty_documents[:5], duplicate]  # 6 docs, 1 duplicate
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_documents(bad_set)
        # Should mention count problem AND duplicate
        assert len(exc_info.value.failures) >= 2


# ---------------------------------------------------------------------------
# validate_chunks
# ---------------------------------------------------------------------------


class TestValidateChunks:
    """Tests for :func:`validate_chunks`."""

    def test_valid_set_passes(
        self,
        one_fifty_chunks: List[Chunk],
        thirty_documents: List[Document],
    ) -> None:
        """Exactly 150 valid chunks must pass."""
        doc_ids = {d.document_id for d in thirty_documents}
        validate_chunks(one_fifty_chunks, doc_ids)  # must not raise

    def test_wrong_count_raises(
        self,
        one_fifty_chunks: List[Chunk],
        thirty_documents: List[Document],
    ) -> None:
        """Fewer than 150 chunks must raise DatasetValidationError."""
        doc_ids = {d.document_id for d in thirty_documents}
        with pytest.raises(DatasetValidationError, match=str(EXPECTED_CHUNK_COUNT)):
            validate_chunks(one_fifty_chunks[:149], doc_ids)

    def test_duplicate_chunk_ids_raises(
        self,
        one_fifty_chunks: List[Chunk],
        thirty_documents: List[Document],
    ) -> None:
        """Duplicate chunk_id values must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        duplicate = one_fifty_chunks[0]
        with pytest.raises(DatasetValidationError, match="Duplicate chunk_id"):
            validate_chunks([*one_fifty_chunks, duplicate], doc_ids)

    def test_orphan_chunk_raises(
        self,
        one_fifty_chunks: List[Chunk],
        thirty_documents: List[Document],
    ) -> None:
        """A chunk referencing a non-existent document_id must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        doc_ids.discard("DOC001")  # remove one document
        with pytest.raises(DatasetValidationError, match="unknown document_id"):
            validate_chunks(one_fifty_chunks, doc_ids)

    def test_empty_doc_ids_raises(self, one_fifty_chunks: List[Chunk]) -> None:
        """All chunks are orphaned when document_ids is empty."""
        with pytest.raises(DatasetValidationError):
            validate_chunks(one_fifty_chunks, set())


# ---------------------------------------------------------------------------
# validate_questions
# ---------------------------------------------------------------------------


class TestValidateQuestions:
    """Tests for :func:`validate_questions`."""

    def test_valid_set_passes(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """Exactly 150 valid questions with correct distribution must pass."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        validate_questions(one_fifty_questions, doc_ids, chunk_ids)  # must not raise

    def test_wrong_count_raises(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """Fewer than 150 questions must raise DatasetValidationError."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        with pytest.raises(DatasetValidationError, match=str(EXPECTED_QUESTION_COUNT)):
            validate_questions(one_fifty_questions[:140], doc_ids, chunk_ids)

    def test_duplicate_question_ids_raises(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """Duplicate question_id values must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        duplicate = one_fifty_questions[0]
        with pytest.raises(DatasetValidationError, match="Duplicate question_id"):
            validate_questions([*one_fifty_questions, duplicate], doc_ids, chunk_ids)

    def test_invalid_document_reference_raises(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """A question referencing a non-existent document must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        doc_ids.discard("DOC001")  # simulate missing document
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        with pytest.raises(DatasetValidationError, match="unknown document_id"):
            validate_questions(one_fifty_questions, doc_ids, chunk_ids)

    def test_invalid_chunk_reference_raises(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """A question referencing a non-existent chunk must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        chunk_ids.discard("DOC001_CHUNK_01")  # simulate missing chunk
        with pytest.raises(DatasetValidationError, match="unknown chunk_id"):
            validate_questions(one_fifty_questions, doc_ids, chunk_ids)

    def test_wrong_question_distribution_raises(
        self,
        one_fifty_questions: List[EvaluationQuestion],
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """A mismatched question type distribution must be flagged."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}

        # Replace first 5 direct questions with semantic ones →
        # direct=20 (expected 25), semantic=30 (expected 25): two violations.
        modified = list(one_fifty_questions)
        for i in range(5):
            q = modified[i]
            modified[i] = EvaluationQuestion(
                question_id=q.question_id,
                question=q.question,
                question_type=QuestionType.SEMANTIC,  # wrong: should be direct
                expected_answer=q.expected_answer,
                relevant_documents=q.relevant_documents,
                relevant_chunks=q.relevant_chunks,
                answerable=q.answerable,
            )

        with pytest.raises(DatasetValidationError, match="direct"):
            validate_questions(modified, doc_ids, chunk_ids)


    def test_empty_collection_raises(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
    ) -> None:
        """An empty question list must raise DatasetValidationError."""
        doc_ids = {d.document_id for d in thirty_documents}
        chunk_ids = {c.chunk_id for c in one_fifty_chunks}
        with pytest.raises(DatasetValidationError):
            validate_questions([], doc_ids, chunk_ids)


# ---------------------------------------------------------------------------
# validate_cross_references
# ---------------------------------------------------------------------------


class TestValidateCrossReferences:
    """Tests for :func:`validate_cross_references`."""

    def test_valid_dataset_passes(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
        one_fifty_questions: List[EvaluationQuestion],
    ) -> None:
        """All cross-references in a valid dataset must pass."""
        doc_ids, chunk_ids = validate_cross_references(
            thirty_documents, one_fifty_chunks, one_fifty_questions
        )
        assert len(doc_ids) == 30
        assert len(chunk_ids) == 150

    def test_orphan_chunk_raises(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
        one_fifty_questions: List[EvaluationQuestion],
    ) -> None:
        """A chunk whose document_id has no matching document must raise."""
        docs_minus_one = thirty_documents[1:]  # remove DOC001
        with pytest.raises(DatasetValidationError, match="missing documents"):
            validate_cross_references(docs_minus_one, one_fifty_chunks, one_fifty_questions)

    def test_returns_id_sets(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
        one_fifty_questions: List[EvaluationQuestion],
    ) -> None:
        """The returned sets must contain the correct IDs."""
        doc_ids, chunk_ids = validate_cross_references(
            thirty_documents, one_fifty_chunks, one_fifty_questions
        )
        assert "DOC001" in doc_ids
        assert "DOC001_CHUNK_01" in chunk_ids


# ---------------------------------------------------------------------------
# validate_full_dataset
# ---------------------------------------------------------------------------


class TestValidateFullDataset:
    """Tests for :func:`validate_full_dataset`."""

    def test_valid_full_dataset_passes(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
        one_fifty_questions: List[EvaluationQuestion],
    ) -> None:
        """A complete, valid dataset must pass without raising."""
        validate_full_dataset(thirty_documents, one_fifty_chunks, one_fifty_questions)

    def test_multiple_failures_collected(
        self,
        thirty_documents: List[Document],
        one_fifty_chunks: List[Chunk],
        one_fifty_questions: List[EvaluationQuestion],
    ) -> None:
        """Failures from multiple validators must all be collected and raised together."""
        # Remove 1 doc (causes chunk orphan + question ref errors + count error)
        docs_minus_one = thirty_documents[1:]
        with pytest.raises(DatasetValidationError) as exc_info:
            validate_full_dataset(docs_minus_one, one_fifty_chunks, one_fifty_questions)
        # Expect multiple failure messages
        assert len(exc_info.value.failures) > 1
