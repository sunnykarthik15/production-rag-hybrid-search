"""
Data loaders for the Nexora RAG dataset.

All loaders:
  - stream JSONL records one-by-one (no bulk ``json.loads`` of the entire file).
  - validate every record with Pydantic and surface descriptive errors.
  - raise ``DataLoadError`` on the first malformed record (fail-fast).

Typical usage
-------------
>>> from pathlib import Path
>>> from app.ingestion.loaders import (
...     load_documents_jsonl,
...     load_document_json,
...     load_chunks_jsonl,
...     load_questions_jsonl,
... )
>>> docs = list(load_documents_jsonl(Path("data/raw/documents.jsonl")))
>>> chunks = list(load_chunks_jsonl(Path("data/processed/chunks.jsonl")))
>>> questions = list(load_questions_jsonl(Path("data/evaluation/questions.jsonl")))
"""

import json
import logging
from pathlib import Path
from typing import Generator

from pydantic import ValidationError

from app.ingestion.models import Chunk, Document, EvaluationQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataLoadError(Exception):
    """Raised when a dataset file cannot be loaded or a record fails validation.

    Attributes
    ----------
    file_path : Path
        The file that triggered the error.
    line_number : int | None
        1-based line number within the file, if applicable.
    original_error : Exception | None
        The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        file_path: Path,
        line_number: int | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.file_path = file_path
        self.line_number = line_number
        self.original_error = original_error
        location = f" (line {line_number})" if line_number is not None else ""
        super().__init__(f"{file_path.name}{location}: {message}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_file(path: Path) -> None:
    """Validate that *path* exists and is a readable file.

    Raises
    ------
    DataLoadError
        If the file does not exist or is not a regular file.
    """
    if not path.exists():
        raise DataLoadError(f"File not found: {path}", file_path=path)
    if not path.is_file():
        raise DataLoadError(f"Path is not a regular file: {path}", file_path=path)


def _parse_jsonl_line(raw: str, line_number: int, path: Path) -> dict:
    """Decode a single JSONL line into a Python dict.

    Parameters
    ----------
    raw : str
        Raw text of the line (may have trailing whitespace / newlines).
    line_number : int
        1-based line position in the file (used for error messages).
    path : Path
        Source file path (used for error messages).

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    DataLoadError
        If the line is not valid JSON.
    """
    stripped = raw.strip()
    if not stripped:
        raise DataLoadError(
            "Encountered an unexpected blank line in JSONL file.",
            file_path=path,
            line_number=line_number,
        )
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise DataLoadError(
            f"Invalid JSON: {exc.msg}",
            file_path=path,
            line_number=line_number,
            original_error=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_documents_jsonl(path: Path) -> Generator[Document, None, None]:
    """Stream and validate every document from a JSONL file.

    Each non-blank line is expected to be a JSON object conforming to the
    :class:`~app.ingestion.models.Document` schema.

    Parameters
    ----------
    path : Path
        Path to the ``documents.jsonl`` file.

    Yields
    ------
    Document
        One validated :class:`~app.ingestion.models.Document` per line.

    Raises
    ------
    DataLoadError
        On file-not-found, invalid JSON, or Pydantic validation failure.
    """
    _open_file(path)
    logger.debug("Loading documents from JSONL: %s", path)

    with path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                logger.debug("Skipping blank line %d in %s", line_number, path.name)
                continue

            data = _parse_jsonl_line(raw_line, line_number, path)

            try:
                doc = Document.model_validate(data)
            except ValidationError as exc:
                raise DataLoadError(
                    f"Document validation failed: {exc}",
                    file_path=path,
                    line_number=line_number,
                    original_error=exc,
                ) from exc

            logger.debug("Loaded document %s", doc.document_id)
            yield doc


def load_document_json(path: Path) -> Document:
    """Load and validate a single document from an individual JSON file.

    Parameters
    ----------
    path : Path
        Path to a ``DOC*.json`` file under ``data/raw/documents/``.

    Returns
    -------
    Document
        Validated :class:`~app.ingestion.models.Document`.

    Raises
    ------
    DataLoadError
        On file-not-found, invalid JSON, or Pydantic validation failure.
    """
    _open_file(path)
    logger.debug("Loading document from JSON: %s", path)

    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise DataLoadError(
            f"Invalid JSON: {exc.msg}",
            file_path=path,
            original_error=exc,
        ) from exc

    try:
        doc = Document.model_validate(data)
    except ValidationError as exc:
        raise DataLoadError(
            f"Document validation failed: {exc}",
            file_path=path,
            original_error=exc,
        ) from exc

    logger.debug("Loaded document %s from %s", doc.document_id, path.name)
    return doc


def load_chunks_jsonl(path: Path) -> Generator[Chunk, None, None]:
    """Stream and validate every chunk from a JSONL file.

    Each non-blank line is expected to be a JSON object conforming to the
    :class:`~app.ingestion.models.Chunk` schema.

    Parameters
    ----------
    path : Path
        Path to the ``chunks.jsonl`` file.

    Yields
    ------
    Chunk
        One validated :class:`~app.ingestion.models.Chunk` per line.

    Raises
    ------
    DataLoadError
        On file-not-found, invalid JSON, or Pydantic validation failure.
    """
    _open_file(path)
    logger.debug("Loading chunks from JSONL: %s", path)

    with path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                logger.debug("Skipping blank line %d in %s", line_number, path.name)
                continue

            data = _parse_jsonl_line(raw_line, line_number, path)

            try:
                chunk = Chunk.model_validate(data)
            except ValidationError as exc:
                raise DataLoadError(
                    f"Chunk validation failed: {exc}",
                    file_path=path,
                    line_number=line_number,
                    original_error=exc,
                ) from exc

            logger.debug("Loaded chunk %s", chunk.chunk_id)
            yield chunk


def load_questions_jsonl(path: Path) -> Generator[EvaluationQuestion, None, None]:
    """Stream and validate every evaluation question from a JSONL file.

    Each non-blank line is expected to be a JSON object conforming to the
    :class:`~app.ingestion.models.EvaluationQuestion` schema.

    Parameters
    ----------
    path : Path
        Path to the ``questions.jsonl`` file.

    Yields
    ------
    EvaluationQuestion
        One validated :class:`~app.ingestion.models.EvaluationQuestion` per line.

    Raises
    ------
    DataLoadError
        On file-not-found, invalid JSON, or Pydantic validation failure.
    """
    _open_file(path)
    logger.debug("Loading evaluation questions from JSONL: %s", path)

    with path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            if not raw_line.strip():
                logger.debug("Skipping blank line %d in %s", line_number, path.name)
                continue

            data = _parse_jsonl_line(raw_line, line_number, path)

            try:
                question = EvaluationQuestion.model_validate(data)
            except ValidationError as exc:
                raise DataLoadError(
                    f"EvaluationQuestion validation failed: {exc}",
                    file_path=path,
                    line_number=line_number,
                    original_error=exc,
                ) from exc

            logger.debug("Loaded question %s", question.question_id)
            yield question
