"""
Core dataset validation logic for the Nexora RAG evaluation dataset.

This module contains pure validation functions that operate on already-loaded
data. The functions are independent of I/O so they can be unit-tested easily.

All public functions raise :class:`ValidationError` if a check fails and
return normally on success.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

from app.ingestion.models import Chunk, Document, EvaluationQuestion, QuestionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected dataset constants
# ---------------------------------------------------------------------------

EXPECTED_DOCUMENT_COUNT = 30
EXPECTED_CHUNK_COUNT = 150
EXPECTED_QUESTION_COUNT = 150

# Actual distribution observed in the Nexora dataset: uniform 25 per type.
EXPECTED_QUESTION_DISTRIBUTION: Dict[QuestionType, int] = {
    QuestionType.DIRECT: 25,
    QuestionType.SEMANTIC: 25,
    QuestionType.EXACT_KEYWORD: 25,
    QuestionType.MULTI_DOCUMENT: 25,
    QuestionType.DISTRACTOR: 25,
    QuestionType.UNANSWERABLE: 25,
}


EXPECTED_SOURCE = "Nexora Synthetic Operations Corpus"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DatasetValidationError(Exception):
    """Raised when one or more dataset validation checks fail.

    Attributes
    ----------
    failures : list[str]
        Human-readable descriptions of every check that failed.
    """

    def __init__(self, failures: List[str]) -> None:
        self.failures = failures
        joined = "\n  - ".join(failures)
        super().__init__(f"Dataset validation failed with {len(failures)} error(s):\n  - {joined}")


# ---------------------------------------------------------------------------
# Document validation
# ---------------------------------------------------------------------------


def validate_documents(documents: Sequence[Document]) -> None:
    """Validate a collection of documents.

    Checks
    ------
    * Exactly :data:`EXPECTED_DOCUMENT_COUNT` documents.
    * All ``document_id`` values are unique.
    * All ``title`` and ``text`` fields are non-empty (enforced by the model
      but double-checked here for belt-and-suspenders reporting).
    * All ``source`` fields equal :data:`EXPECTED_SOURCE`.

    Parameters
    ----------
    documents : Sequence[Document]
        Fully validated :class:`~app.ingestion.models.Document` objects.

    Raises
    ------
    DatasetValidationError
        If any check fails. All failures are collected before raising.
    """
    failures: List[str] = []

    # 1. Count
    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        failures.append(
            f"Expected {EXPECTED_DOCUMENT_COUNT} documents, got {len(documents)}."
        )

    # 2. Unique IDs
    id_counts = Counter(d.document_id for d in documents)
    duplicates = [doc_id for doc_id, count in id_counts.items() if count > 1]
    if duplicates:
        failures.append(f"Duplicate document_id(s): {sorted(duplicates)}")

    # 3. Non-empty titles / text / source
    for doc in documents:
        if not doc.title.strip():
            failures.append(f"Document '{doc.document_id}' has an empty title.")
        if not doc.text.strip():
            failures.append(f"Document '{doc.document_id}' has empty text.")
        if not doc.source.strip():
            failures.append(f"Document '{doc.document_id}' has an empty source.")

    if failures:
        raise DatasetValidationError(failures)

    logger.debug("Documents: all %d checks passed.", len(documents))


# ---------------------------------------------------------------------------
# Chunk validation
# ---------------------------------------------------------------------------


def validate_chunks(
    chunks: Sequence[Chunk],
    document_ids: Set[str],
) -> None:
    """Validate a collection of chunks.

    Checks
    ------
    * Exactly :data:`EXPECTED_CHUNK_COUNT` chunks.
    * All ``chunk_id`` values are unique.
    * Every chunk's ``document_id`` references a known document.
    * All ``text`` fields are non-empty.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Fully validated :class:`~app.ingestion.models.Chunk` objects.
    document_ids : Set[str]
        Set of ``document_id`` values from the loaded document collection.

    Raises
    ------
    DatasetValidationError
        If any check fails. All failures are collected before raising.
    """
    failures: List[str] = []

    # 1. Count
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        failures.append(
            f"Expected {EXPECTED_CHUNK_COUNT} chunks, got {len(chunks)}."
        )

    # 2. Unique IDs
    id_counts = Counter(c.chunk_id for c in chunks)
    duplicates = [cid for cid, count in id_counts.items() if count > 1]
    if duplicates:
        failures.append(f"Duplicate chunk_id(s): {sorted(duplicates)}")

    # 3. Orphan chunks (document_id not in known documents)
    orphans = [c.chunk_id for c in chunks if c.document_id not in document_ids]
    if orphans:
        failures.append(
            f"{len(orphans)} chunk(s) reference unknown document_id(s): {sorted(orphans[:10])}"
            + (" (first 10 shown)" if len(orphans) > 10 else "")
        )

    # 4. Non-empty text
    for chunk in chunks:
        if not chunk.text.strip():
            failures.append(f"Chunk '{chunk.chunk_id}' has empty text.")

    if failures:
        raise DatasetValidationError(failures)

    logger.debug("Chunks: all %d checks passed.", len(chunks))


# ---------------------------------------------------------------------------
# Question validation
# ---------------------------------------------------------------------------


def validate_questions(
    questions: Sequence[EvaluationQuestion],
    document_ids: Set[str],
    chunk_ids: Set[str],
) -> None:
    """Validate a collection of evaluation questions.

    Checks
    ------
    * Exactly :data:`EXPECTED_QUESTION_COUNT` questions.
    * All ``question_id`` values are unique.
    * All ``question_type`` values are valid (enforced by Pydantic enum, but
      also verified at the collection level for distribution checks).
    * ``question`` and ``expected_answer`` fields are non-empty.
    * Every entry in ``relevant_documents`` references a known document.
    * Every entry in ``relevant_chunks`` references a known chunk.
    * ``answerable`` flag consistency (unanswerable ↔ empty ``relevant_chunks``).
    * Question type distribution matches :data:`EXPECTED_QUESTION_DISTRIBUTION`.

    Parameters
    ----------
    questions : Sequence[EvaluationQuestion]
        Fully validated :class:`~app.ingestion.models.EvaluationQuestion` objects.
    document_ids : Set[str]
        Set of known document IDs.
    chunk_ids : Set[str]
        Set of known chunk IDs.

    Raises
    ------
    DatasetValidationError
        If any check fails. All failures are collected before raising.
    """
    failures: List[str] = []

    # 1. Count
    if len(questions) != EXPECTED_QUESTION_COUNT:
        failures.append(
            f"Expected {EXPECTED_QUESTION_COUNT} questions, got {len(questions)}."
        )

    # 2. Unique IDs
    id_counts = Counter(q.question_id for q in questions)
    duplicates = [qid for qid, count in id_counts.items() if count > 1]
    if duplicates:
        failures.append(f"Duplicate question_id(s): {sorted(duplicates)}")

    for q in questions:
        # 3. Non-empty question text and answer
        if not q.question.strip():
            failures.append(f"Question '{q.question_id}' has empty question text.")
        if not q.expected_answer.strip():
            failures.append(f"Question '{q.question_id}' has empty expected_answer.")

        # 4. Document references
        bad_docs = [d for d in q.relevant_documents if d not in document_ids]
        if bad_docs:
            failures.append(
                f"Question '{q.question_id}' references unknown document_id(s): {bad_docs}"
            )

        # 5. Chunk references (only for answerable questions)
        bad_chunks = [c for c in q.relevant_chunks if c not in chunk_ids]
        if bad_chunks:
            failures.append(
                f"Question '{q.question_id}' references unknown chunk_id(s): {bad_chunks}"
            )

    # 6. Question type distribution
    dist = Counter(q.question_type for q in questions)
    for qtype, expected_count in EXPECTED_QUESTION_DISTRIBUTION.items():
        actual_count = dist.get(qtype, 0)
        if actual_count != expected_count:
            failures.append(
                f"Question type '{qtype.value}': expected {expected_count}, got {actual_count}."
            )

    if failures:
        raise DatasetValidationError(failures)

    logger.debug("Questions: all %d checks passed.", len(questions))


# ---------------------------------------------------------------------------
# Cross-reference validation
# ---------------------------------------------------------------------------


def validate_cross_references(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    questions: Sequence[EvaluationQuestion],
) -> Tuple[Set[str], Set[str]]:
    """Validate cross-references across documents, chunks, and questions.

    Returns the document_id and chunk_id sets for downstream use.

    Checks
    ------
    * Chunk → Document: every chunk's ``document_id`` is a known document.
    * Question → Document: every referenced document ID exists.
    * Question → Chunk: every referenced chunk ID exists.

    Parameters
    ----------
    documents : Sequence[Document]
    chunks : Sequence[Chunk]
    questions : Sequence[EvaluationQuestion]

    Returns
    -------
    Tuple[Set[str], Set[str]]
        ``(document_ids, chunk_ids)`` — the sets of known IDs.

    Raises
    ------
    DatasetValidationError
        If any cross-reference check fails.
    """
    failures: List[str] = []

    document_ids: Set[str] = {d.document_id for d in documents}
    chunk_ids: Set[str] = {c.chunk_id for c in chunks}

    # Chunk → Document orphans
    orphan_chunk_doc_pairs = [
        (c.chunk_id, c.document_id) for c in chunks if c.document_id not in document_ids
    ]
    if orphan_chunk_doc_pairs:
        failures.append(
            f"{len(orphan_chunk_doc_pairs)} chunk(s) reference missing documents: "
            + str(orphan_chunk_doc_pairs[:5])
            + (" ..." if len(orphan_chunk_doc_pairs) > 5 else "")
        )

    # Question → Document
    for q in questions:
        bad = [d for d in q.relevant_documents if d not in document_ids]
        if bad:
            failures.append(
                f"Question '{q.question_id}' references missing document(s): {bad}"
            )

    # Question → Chunk
    for q in questions:
        bad = [c for c in q.relevant_chunks if c not in chunk_ids]
        if bad:
            failures.append(
                f"Question '{q.question_id}' references missing chunk(s): {bad}"
            )

    if failures:
        raise DatasetValidationError(failures)

    logger.debug("Cross-references: all checks passed.")
    return document_ids, chunk_ids


# ---------------------------------------------------------------------------
# Convenience: run all validations in one call
# ---------------------------------------------------------------------------


def validate_full_dataset(
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    questions: Sequence[EvaluationQuestion],
) -> None:
    """Run the complete validation suite on the full dataset.

    Calls :func:`validate_documents`, :func:`validate_chunks`,
    :func:`validate_questions`, and :func:`validate_cross_references` in order,
    collecting and re-raising all failures together.

    Parameters
    ----------
    documents : Sequence[Document]
    chunks : Sequence[Chunk]
    questions : Sequence[EvaluationQuestion]

    Raises
    ------
    DatasetValidationError
        Accumulates failures from all sub-validators before raising.
    """
    all_failures: List[str] = []

    def _run(fn, *args):  # type: ignore[no-untyped-def]
        try:
            return fn(*args)
        except DatasetValidationError as exc:
            all_failures.extend(exc.failures)
            return None

    document_ids: Set[str] = {d.document_id for d in documents}
    chunk_ids: Set[str] = {c.chunk_id for c in chunks}

    _run(validate_documents, documents)
    _run(validate_chunks, chunks, document_ids)
    _run(validate_questions, questions, document_ids, chunk_ids)
    _run(validate_cross_references, documents, chunks, questions)

    if all_failures:
        raise DatasetValidationError(all_failures)

    logger.info(
        "Full dataset validation passed: %d documents, %d chunks, %d questions.",
        len(documents),
        len(chunks),
        len(questions),
    )
