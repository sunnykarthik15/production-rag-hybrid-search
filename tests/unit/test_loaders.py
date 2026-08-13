"""Unit tests for the dataset loaders (loaders.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from app.ingestion.loaders import (
    DataLoadError,
    load_chunks_jsonl,
    load_document_json,
    load_documents_jsonl,
    load_questions_jsonl,
)
from app.ingestion.models import Chunk, Document, EvaluationQuestion


# ---------------------------------------------------------------------------
# load_documents_jsonl
# ---------------------------------------------------------------------------


class TestLoadDocumentsJsonl:
    """Tests for :func:`load_documents_jsonl`."""

    def test_loads_all_documents(
        self, tmp_documents_jsonl: Path, thirty_documents: List[Document]
    ) -> None:
        """All 30 documents must be loaded and validated."""
        result = list(load_documents_jsonl(tmp_documents_jsonl))
        assert len(result) == 30
        assert all(isinstance(d, Document) for d in result)

    def test_document_ids_preserved(
        self, tmp_documents_jsonl: Path, thirty_documents: List[Document]
    ) -> None:
        """Loaded document IDs must match the fixture IDs."""
        result = list(load_documents_jsonl(tmp_documents_jsonl))
        ids = {d.document_id for d in result}
        expected_ids = {d.document_id for d in thirty_documents}
        assert ids == expected_ids

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A missing file must raise DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            list(load_documents_jsonl(tmp_path / "nonexistent.jsonl"))

    def test_malformed_json_raises(self, tmp_malformed_jsonl: Path) -> None:
        """A line with invalid JSON must raise DataLoadError."""
        with pytest.raises(DataLoadError):
            list(load_documents_jsonl(tmp_malformed_jsonl))

    def test_invalid_document_record_raises(self, tmp_path: Path) -> None:
        """A valid JSON line that fails schema validation must raise DataLoadError."""
        path = tmp_path / "bad_doc.jsonl"
        # Missing required fields
        path.write_text(json.dumps({"document_id": "DOC001"}) + "\n", encoding="utf-8")
        with pytest.raises(DataLoadError, match="validation"):
            list(load_documents_jsonl(path))

    def test_blank_trailing_line_is_skipped(
        self, tmp_path: Path, thirty_documents: List[Document]
    ) -> None:
        """A trailing blank line at EOF must be silently skipped."""
        path = tmp_path / "docs_trailing.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for doc in thirty_documents:
                fh.write(doc.model_dump_json() + "\n")
            fh.write("\n")  # trailing blank line
        result = list(load_documents_jsonl(path))
        assert len(result) == 30

    def test_error_contains_line_number(self, tmp_path: Path) -> None:
        """DataLoadError for a bad line must include the line number."""
        path = tmp_path / "bad_line.jsonl"
        # Line 3 is malformed
        path.write_text(
            '{"chunk_id":"C1"}\nNOT_JSON\n',
            encoding="utf-8",
        )
        with pytest.raises(DataLoadError) as exc_info:
            list(load_documents_jsonl(path))
        # Error message should reference a line number
        assert exc_info.value.line_number is not None


# ---------------------------------------------------------------------------
# load_document_json
# ---------------------------------------------------------------------------


class TestLoadDocumentJson:
    """Tests for :func:`load_document_json`."""

    def test_loads_single_document(self, tmp_path: Path, valid_document: Document) -> None:
        """A single DOC*.json file must load and validate correctly."""
        path = tmp_path / "DOC001.json"
        path.write_text(valid_document.model_dump_json(), encoding="utf-8")

        result = load_document_json(path)
        assert isinstance(result, Document)
        assert result.document_id == valid_document.document_id

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A missing file must raise DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            load_document_json(tmp_path / "DOC999.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """An invalid JSON file must raise DataLoadError."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}", encoding="utf-8")
        with pytest.raises(DataLoadError):
            load_document_json(path)

    def test_invalid_schema_raises(self, tmp_path: Path) -> None:
        """A JSON file with wrong schema must raise DataLoadError."""
        path = tmp_path / "wrong_schema.json"
        path.write_text(json.dumps({"document_id": "DOC001"}), encoding="utf-8")
        with pytest.raises(DataLoadError, match="validation"):
            load_document_json(path)


# ---------------------------------------------------------------------------
# load_chunks_jsonl
# ---------------------------------------------------------------------------


class TestLoadChunksJsonl:
    """Tests for :func:`load_chunks_jsonl`."""

    def test_loads_all_chunks(
        self, tmp_chunks_jsonl: Path, one_fifty_chunks: List[Chunk]
    ) -> None:
        """All 150 chunks must be loaded and validated."""
        result = list(load_chunks_jsonl(tmp_chunks_jsonl))
        assert len(result) == 150
        assert all(isinstance(c, Chunk) for c in result)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A missing file must raise DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            list(load_chunks_jsonl(tmp_path / "nonexistent.jsonl"))

    def test_malformed_json_raises(self, tmp_malformed_jsonl: Path) -> None:
        """A line with invalid JSON must raise DataLoadError."""
        with pytest.raises(DataLoadError):
            list(load_chunks_jsonl(tmp_malformed_jsonl))

    def test_invalid_chunk_record_raises(self, tmp_path: Path) -> None:
        """A valid JSON line that fails Chunk schema validation must raise DataLoadError."""
        path = tmp_path / "bad_chunk.jsonl"
        path.write_text(json.dumps({"chunk_id": "C1"}) + "\n", encoding="utf-8")
        with pytest.raises(DataLoadError, match="validation"):
            list(load_chunks_jsonl(path))


# ---------------------------------------------------------------------------
# load_questions_jsonl
# ---------------------------------------------------------------------------


class TestLoadQuestionsJsonl:
    """Tests for :func:`load_questions_jsonl`."""

    def test_loads_all_questions(
        self, tmp_questions_jsonl: Path, one_fifty_questions: List[EvaluationQuestion]
    ) -> None:
        """All 150 questions must be loaded and validated."""
        result = list(load_questions_jsonl(tmp_questions_jsonl))
        assert len(result) == 150
        assert all(isinstance(q, EvaluationQuestion) for q in result)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """A missing file must raise DataLoadError."""
        with pytest.raises(DataLoadError, match="not found"):
            list(load_questions_jsonl(tmp_path / "nonexistent.jsonl"))

    def test_malformed_json_raises(self, tmp_malformed_jsonl: Path) -> None:
        """A line with invalid JSON must raise DataLoadError."""
        with pytest.raises(DataLoadError):
            list(load_questions_jsonl(tmp_malformed_jsonl))

    def test_invalid_question_type_raises(self, tmp_path: Path) -> None:
        """An invalid question_type in the JSON must raise DataLoadError."""
        path = tmp_path / "bad_qtype.jsonl"
        bad_record = {
            "question_id": "Q001",
            "question": "Some question?",
            "question_type": "totally_invalid",
            "expected_answer": "answer",
            "relevant_documents": ["DOC001"],
            "relevant_chunks": ["DOC001_CHUNK_01"],
            "answerable": True,
        }
        path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")
        with pytest.raises(DataLoadError, match="validation"):
            list(load_questions_jsonl(path))
