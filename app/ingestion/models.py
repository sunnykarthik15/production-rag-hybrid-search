"""
Pydantic data models for the Nexora RAG dataset.

Schemas are derived from the actual dataset files:
  - data/raw/documents.jsonl  (and data/raw/documents/DOC*.json)
  - data/processed/chunks.jsonl
  - data/evaluation/questions.jsonl
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    """Valid question type values present in the Nexora evaluation set."""

    DIRECT = "direct"
    SEMANTIC = "semantic"
    EXACT_KEYWORD = "exact_keyword"
    MULTI_DOCUMENT = "multi_document"
    DISTRACTOR = "distractor"
    UNANSWERABLE = "unanswerable"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Represents a single source document from the Nexora corpus.

    Corresponds to one JSON object in ``data/raw/documents.jsonl`` or an
    individual ``data/raw/documents/DOC*.json`` file.
    """

    document_id: str = Field(..., description="Unique document identifier, e.g. 'DOC001'.")
    title: str = Field(..., description="Full document title.")
    department: str = Field(..., description="Responsible department.")
    topic: str = Field(..., description="Primary topic covered by this document.")
    owner: str = Field(..., description="Team or person responsible for the module.")
    module_code: str = Field(..., description="Module identifier code, e.g. 'NX-FAC-100'.")
    text: str = Field(..., description="Full document body text.")
    source: str = Field(..., description="Corpus name, e.g. 'Nexora Synthetic Operations Corpus'.")
    version: str = Field(..., description="Dataset version string, e.g. '1.0'.")

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        """Ensure document_id is a non-empty string."""
        if not v.strip():
            raise ValueError("document_id must not be empty.")
        return v

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Ensure title is a non-empty string."""
        if not v.strip():
            raise ValueError("title must not be empty.")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure text body is non-empty."""
        if not v.strip():
            raise ValueError("text must not be empty.")
        return v

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        """Ensure source field is non-empty."""
        if not v.strip():
            raise ValueError("source must not be empty.")
        return v


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """Represents a single text chunk produced during document processing.

    Corresponds to one JSON object in ``data/processed/chunks.jsonl``.
    """

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier, e.g. 'DOC001_CHUNK_01'.",
    )
    document_id: str = Field(
        ...,
        description="ID of the parent document this chunk belongs to.",
    )
    title: str = Field(..., description="Title of the parent document.")
    department: str = Field(..., description="Department associated with the parent document.")
    text: str = Field(..., description="Text content of this chunk.")
    chunk_index: int = Field(..., ge=1, description="1-based position of this chunk within its document.")
    source: str = Field(..., description="Corpus name inherited from the source document.")

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_id must not be empty.")
        return v

    @field_validator("document_id")
    @classmethod
    def document_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty.")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty.")
        return v


# ---------------------------------------------------------------------------
# EvaluationQuestion
# ---------------------------------------------------------------------------


class EvaluationQuestion(BaseModel):
    """Represents a single evaluation question.

    Corresponds to one JSON object in ``data/evaluation/questions.jsonl``.

    Notes
    -----
    * ``relevant_chunks`` may be an empty list for ``unanswerable`` questions.
    * ``expected_answer`` is always non-empty (even for unanswerable questions
      the dataset provides an explanation string).
    """

    question_id: str = Field(..., description="Unique question identifier, e.g. 'Q001'.")
    question: str = Field(..., description="The natural-language question text.")
    question_type: QuestionType = Field(..., description="Category of the question.")
    expected_answer: str = Field(..., description="Ground-truth answer string.")
    relevant_documents: List[str] = Field(
        ...,
        description="List of document IDs that contain the answer.",
    )
    relevant_chunks: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs that directly answer the question. "
        "May be empty for unanswerable questions.",
    )
    answerable: bool = Field(
        ...,
        description="Whether the question can be answered from the supplied corpus.",
    )

    @field_validator("question_id")
    @classmethod
    def question_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question_id must not be empty.")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question text must not be empty.")
        return v

    @field_validator("expected_answer")
    @classmethod
    def expected_answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("expected_answer must not be empty.")
        return v

    @field_validator("relevant_documents")
    @classmethod
    def relevant_documents_not_empty(cls, v: List[str]) -> List[str]:
        """Every question must reference at least one document."""
        if not v:
            raise ValueError("relevant_documents must contain at least one document ID.")
        return v

    @model_validator(mode="after")
    def unanswerable_has_empty_chunks(self) -> "EvaluationQuestion":
        """Cross-field rule: unanswerable questions must have empty relevant_chunks."""
        if self.question_type == QuestionType.UNANSWERABLE and self.relevant_chunks:
            raise ValueError(
                f"Question '{self.question_id}' is unanswerable but has non-empty "
                f"relevant_chunks: {self.relevant_chunks}."
            )
        if self.question_type != QuestionType.UNANSWERABLE and not self.relevant_chunks:
            raise ValueError(
                f"Answerable question '{self.question_id}' (type={self.question_type}) "
                f"must have at least one relevant_chunk."
            )
        return self
