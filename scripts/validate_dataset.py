#!/usr/bin/env python3
"""
scripts/validate_dataset.py
============================
CLI tool to validate the Nexora RAG evaluation dataset.

Usage
-----
    python scripts/validate_dataset.py

Exit codes
----------
    0 — all validation checks passed
    1 — one or more validation checks failed
    2 — dataset file(s) not found or could not be loaded

The script resolves all paths relative to the repository root (the parent
directory of ``scripts/``), so it works correctly from any working directory
inside the project.
"""

from __future__ import annotations

import io
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError with ✓/✗
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Path bootstrap — ensure the project root is on sys.path so that
# ``app.ingestion`` is importable regardless of where the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ingestion.loaders import (  # noqa: E402
    DataLoadError,
    load_chunks_jsonl,
    load_documents_jsonl,
    load_questions_jsonl,
)
from app.ingestion.models import QuestionType  # noqa: E402
from app.ingestion.validator import (  # noqa: E402
    EXPECTED_CHUNK_COUNT,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_QUESTION_COUNT,
    EXPECTED_QUESTION_DISTRIBUTION,
    DatasetValidationError,
    validate_chunks,
    validate_cross_references,
    validate_documents,
    validate_questions,
)

# ---------------------------------------------------------------------------
# Logging — INFO to stdout; suppress noisy debug output during normal runs.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_ROOT = _REPO_ROOT / "data"
DOCUMENTS_JSONL = DATA_ROOT / "raw" / "documents.jsonl"
CHUNKS_JSONL = DATA_ROOT / "processed" / "chunks.jsonl"
QUESTIONS_JSONL = DATA_ROOT / "evaluation" / "questions.jsonl"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
SEP = "=" * 40


def _check_mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _print_section(title: str) -> None:
    print(f"\n{SEP}")
    print(title)
    print(SEP)


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_validation() -> int:  # noqa: PLR0912 PLR0915
    """Execute the full validation suite and print a structured report.

    Returns
    -------
    int
        ``0`` on success, ``1`` on validation failure, ``2`` on load failure.
    """
    _print_section("NEXORA DATASET VALIDATION")

    # ------------------------------------------------------------------
    # Load all data
    # ------------------------------------------------------------------
    print("\nLoading dataset files...")

    for path, label in [
        (DOCUMENTS_JSONL, "documents.jsonl"),
        (CHUNKS_JSONL, "chunks.jsonl"),
        (QUESTIONS_JSONL, "questions.jsonl"),
    ]:
        if not path.exists():
            print(f"  ERROR: File not found: {path}")
            print(f"\n  Expected path: {path}")
            print("  Run this script from the repository root or ensure the dataset")
            print("  has been copied into data/ with the correct structure.")
            return 2

    try:
        documents = list(load_documents_jsonl(DOCUMENTS_JSONL))
        chunks = list(load_chunks_jsonl(CHUNKS_JSONL))
        questions = list(load_questions_jsonl(QUESTIONS_JSONL))
    except DataLoadError as exc:
        print(f"\n  LOAD ERROR: {exc}")
        return 2

    # ------------------------------------------------------------------
    # Per-collection validation (collect all failures, don't short-circuit)
    # ------------------------------------------------------------------
    doc_failures: List[str] = []
    chunk_failures: List[str] = []
    question_failures: List[str] = []
    xref_failures: List[str] = []

    document_ids = {d.document_id for d in documents}
    chunk_ids = {c.chunk_id for c in chunks}

    try:
        validate_documents(documents)
    except DatasetValidationError as exc:
        doc_failures = exc.failures

    try:
        validate_chunks(chunks, document_ids)
    except DatasetValidationError as exc:
        chunk_failures = exc.failures

    try:
        validate_questions(questions, document_ids, chunk_ids)
    except DatasetValidationError as exc:
        question_failures = exc.failures

    try:
        validate_cross_references(documents, chunks, questions)
    except DatasetValidationError as exc:
        xref_failures = exc.failures

    # Duplicate ID checks (summary booleans for display)
    doc_dups = len({d.document_id for d in documents}) < len(documents)
    chunk_dups = len({c.chunk_id for c in chunks}) < len(chunks)
    question_dups = len({q.question_id for q in questions}) < len(questions)
    any_duplicates = doc_dups or chunk_dups or question_dups

    malformed_errors = doc_failures + chunk_failures + question_failures
    any_malformed = bool(malformed_errors)

    all_failures = doc_failures + chunk_failures + question_failures + xref_failures

    # ------------------------------------------------------------------
    # Print summary report
    # ------------------------------------------------------------------
    print()
    doc_ok = len(documents) == EXPECTED_DOCUMENT_COUNT and not doc_failures
    chunk_ok = len(chunks) == EXPECTED_CHUNK_COUNT and not chunk_failures
    question_ok = len(questions) == EXPECTED_QUESTION_COUNT and not question_failures

    print(f"Documents: {len(documents)}/{EXPECTED_DOCUMENT_COUNT} {_check_mark(doc_ok)}")
    print(f"Chunks:    {len(chunks)}/{EXPECTED_CHUNK_COUNT} {_check_mark(chunk_ok)}")
    print(f"Questions: {len(questions)}/{EXPECTED_QUESTION_COUNT} {_check_mark(question_ok)}")

    # Question distribution
    dist: Dict[QuestionType, int] = Counter(q.question_type for q in questions)
    print("\nQuestion distribution:")
    max_name_len = max(len(qt.value) for qt in EXPECTED_QUESTION_DISTRIBUTION)
    for qtype, expected in EXPECTED_QUESTION_DISTRIBUTION.items():
        actual = dist.get(qtype, 0)
        ok = actual == expected
        name_padded = qtype.value.ljust(max_name_len)
        print(f"  {name_padded}  {actual} {_check_mark(ok)}")

    # Cross-references
    xref_ok = not xref_failures
    print(f"\nCross references:  {'PASS ' + _check_mark(True) if xref_ok else 'FAIL ' + _check_mark(False)}")
    print(f"Duplicate IDs:     {'PASS ' + _check_mark(True) if not any_duplicates else 'FAIL ' + _check_mark(False)}")
    print(f"Malformed records: {'PASS ' + _check_mark(True) if not any_malformed else 'FAIL ' + _check_mark(False)}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    if all_failures:
        _print_section("DATASET VALIDATION FAILED")
        print(f"\n{len(all_failures)} error(s) detected:\n")
        for i, failure in enumerate(all_failures, start=1):
            print(f"  {i:>3}. {failure}")
        print()
        return 1

    _print_section("DATASET VALIDATION PASSED")
    print()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_validation())
