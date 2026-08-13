"""Unit tests for Pydantic data models (Document, Chunk, EvaluationQuestion)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.models import Chunk, Document, EvaluationQuestion, QuestionType


# ---------------------------------------------------------------------------
# Document model tests
# ---------------------------------------------------------------------------


class TestDocumentModel:
    """Tests for the Document Pydantic model."""

    def test_valid_document_passes(self, valid_document: Document) -> None:
        """A fully populated, valid document must parse without errors."""
        assert valid_document.document_id == "DOC001"
        assert valid_document.title == "Test Document"
        assert valid_document.source == "Nexora Synthetic Operations Corpus"

    def test_document_id_required(self) -> None:
        """Omitting document_id must raise ValidationError."""
        with pytest.raises(ValidationError, match="document_id"):
            Document(
                title="T",
                department="D",
                topic="t",
                owner="O",
                module_code="NX-X-001",
                text="body",
                source="src",
                version="1.0",
            )

    def test_empty_title_raises(self) -> None:
        """An empty title string must fail validation."""
        with pytest.raises(ValidationError, match="title"):
            Document(
                document_id="DOC001",
                title="   ",  # whitespace only
                department="D",
                topic="t",
                owner="O",
                module_code="NX-X-001",
                text="body",
                source="src",
                version="1.0",
            )

    def test_empty_text_raises(self) -> None:
        """An empty text field must fail validation."""
        with pytest.raises(ValidationError, match="text"):
            Document(
                document_id="DOC001",
                title="Title",
                department="D",
                topic="t",
                owner="O",
                module_code="NX-X-001",
                text="",
                source="src",
                version="1.0",
            )

    def test_empty_source_raises(self) -> None:
        """An empty source field must fail validation."""
        with pytest.raises(ValidationError, match="source"):
            Document(
                document_id="DOC001",
                title="Title",
                department="D",
                topic="t",
                owner="O",
                module_code="NX-X-001",
                text="body",
                source="   ",
                version="1.0",
            )

    def test_model_fields_accessible(self, valid_document: Document) -> None:
        """All expected fields must be accessible as attributes."""
        assert valid_document.department == "Testing"
        assert valid_document.topic == "unit tests"
        assert valid_document.owner == "Test Team"
        assert valid_document.module_code == "NX-TST-001"
        assert valid_document.version == "1.0"


# ---------------------------------------------------------------------------
# Chunk model tests
# ---------------------------------------------------------------------------


class TestChunkModel:
    """Tests for the Chunk Pydantic model."""

    def test_valid_chunk_passes(self, valid_chunk: Chunk) -> None:
        """A fully populated, valid chunk must parse without errors."""
        assert valid_chunk.chunk_id == "DOC001_CHUNK_01"
        assert valid_chunk.document_id == "DOC001"
        assert valid_chunk.chunk_index == 1

    def test_chunk_id_required(self) -> None:
        """Omitting chunk_id must raise ValidationError."""
        with pytest.raises(ValidationError, match="chunk_id"):
            Chunk(
                document_id="DOC001",
                title="T",
                department="D",
                text="body",
                chunk_index=1,
                source="src",
            )

    def test_empty_chunk_text_raises(self) -> None:
        """An empty text field must fail validation."""
        with pytest.raises(ValidationError, match="text"):
            Chunk(
                chunk_id="DOC001_CHUNK_01",
                document_id="DOC001",
                title="T",
                department="D",
                text="   ",  # whitespace only
                chunk_index=1,
                source="src",
            )

    def test_chunk_index_must_be_positive(self) -> None:
        """chunk_index < 1 must fail validation."""
        with pytest.raises(ValidationError, match="chunk_index"):
            Chunk(
                chunk_id="DOC001_CHUNK_01",
                document_id="DOC001",
                title="T",
                department="D",
                text="body",
                chunk_index=0,  # invalid: must be >= 1
                source="src",
            )

    def test_empty_document_id_raises(self) -> None:
        """An empty document_id must fail validation."""
        with pytest.raises(ValidationError, match="document_id"):
            Chunk(
                chunk_id="DOC001_CHUNK_01",
                document_id="   ",
                title="T",
                department="D",
                text="body",
                chunk_index=1,
                source="src",
            )


# ---------------------------------------------------------------------------
# EvaluationQuestion model tests
# ---------------------------------------------------------------------------


class TestEvaluationQuestionModel:
    """Tests for the EvaluationQuestion Pydantic model."""

    def test_valid_answerable_question_passes(self, valid_question: EvaluationQuestion) -> None:
        """A fully populated, valid answerable question must parse without errors."""
        assert valid_question.question_id == "Q001"
        assert valid_question.answerable is True
        assert valid_question.question_type == QuestionType.DIRECT

    def test_valid_unanswerable_question_passes(self) -> None:
        """An unanswerable question with empty relevant_chunks must pass."""
        q = EvaluationQuestion(
            question_id="Q999",
            question="What is the budget?",
            question_type=QuestionType.UNANSWERABLE,
            expected_answer="The corpus does not contain this information.",
            relevant_documents=["DOC001"],
            relevant_chunks=[],
            answerable=False,
        )
        assert q.answerable is False
        assert q.relevant_chunks == []

    def test_invalid_question_type_raises(self) -> None:
        """An unrecognised question_type must fail validation."""
        with pytest.raises(ValidationError, match="question_type"):
            EvaluationQuestion(
                question_id="Q001",
                question="Some question?",
                question_type="invalid_type",  # type: ignore[arg-type]
                expected_answer="answer",
                relevant_documents=["DOC001"],
                relevant_chunks=["DOC001_CHUNK_01"],
                answerable=True,
            )

    def test_empty_question_text_raises(self) -> None:
        """An empty question string must fail validation."""
        with pytest.raises(ValidationError, match="question"):
            EvaluationQuestion(
                question_id="Q001",
                question="   ",  # whitespace only
                question_type=QuestionType.DIRECT,
                expected_answer="answer",
                relevant_documents=["DOC001"],
                relevant_chunks=["DOC001_CHUNK_01"],
                answerable=True,
            )

    def test_empty_expected_answer_raises(self) -> None:
        """An empty expected_answer must fail validation."""
        with pytest.raises(ValidationError, match="expected_answer"):
            EvaluationQuestion(
                question_id="Q001",
                question="Some question?",
                question_type=QuestionType.DIRECT,
                expected_answer="",
                relevant_documents=["DOC001"],
                relevant_chunks=["DOC001_CHUNK_01"],
                answerable=True,
            )

    def test_empty_relevant_documents_raises(self) -> None:
        """relevant_documents must not be empty."""
        with pytest.raises(ValidationError, match="relevant_documents"):
            EvaluationQuestion(
                question_id="Q001",
                question="Some question?",
                question_type=QuestionType.DIRECT,
                expected_answer="answer",
                relevant_documents=[],
                relevant_chunks=["DOC001_CHUNK_01"],
                answerable=True,
            )

    def test_unanswerable_with_chunks_raises(self) -> None:
        """An unanswerable question with non-empty relevant_chunks must fail."""
        with pytest.raises(ValidationError, match="unanswerable"):
            EvaluationQuestion(
                question_id="Q001",
                question="What is the budget?",
                question_type=QuestionType.UNANSWERABLE,
                expected_answer="Not in corpus.",
                relevant_documents=["DOC001"],
                relevant_chunks=["DOC001_CHUNK_01"],  # must be empty for unanswerable
                answerable=False,
            )

    def test_answerable_without_chunks_raises(self) -> None:
        """An answerable (non-unanswerable) question with no chunks must fail."""
        with pytest.raises(ValidationError, match="relevant_chunk"):
            EvaluationQuestion(
                question_id="Q001",
                question="What is the SLA?",
                question_type=QuestionType.DIRECT,
                expected_answer="4 hours",
                relevant_documents=["DOC001"],
                relevant_chunks=[],  # must be non-empty for answerable questions
                answerable=True,
            )

    def test_all_question_types_accepted(self) -> None:
        """Every valid QuestionType enum value must be accepted."""
        for qtype in QuestionType:
            is_unanswerable = qtype == QuestionType.UNANSWERABLE
            q = EvaluationQuestion(
                question_id="Q001",
                question="Test question?",
                question_type=qtype,
                expected_answer="Some answer.",
                relevant_documents=["DOC001"],
                relevant_chunks=[] if is_unanswerable else ["DOC001_CHUNK_01"],
                answerable=not is_unanswerable,
            )
            assert q.question_type == qtype
