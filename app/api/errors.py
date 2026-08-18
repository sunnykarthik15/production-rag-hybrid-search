"""Standardised API error handling for Phase 8 FastAPI application.

Provides custom exception classes, structured error response formatting,
and FastAPI exception handler registrations that prevent leakage of
internal details, tracebacks, and secrets.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.rag.service import RAGError
from app.reranking.service import RerankerError

logger = logging.getLogger(__name__)

# Pattern to detect and redact potential secrets in error messages
_SECRET_PATTERNS = [
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(OPENAI_API_KEY\s*=\s*\S+)", re.IGNORECASE),
    re.compile(r"(Bearer\s+\S+)", re.IGNORECASE),
    re.compile(r"([a-f0-9]{32,})", re.IGNORECASE),
]

# Pattern to detect absolute filesystem paths (Windows drive letters or POSIX roots)
_PATH_PATTERN = re.compile(
    r"[A-Z]:[/\\][^\s\"']+|/(?:home|usr|var|tmp|etc|opt|app)[/\\]?[^\s\"']*",
    re.IGNORECASE,
)


def _sanitize_message(message: str) -> str:
    """Remove secrets, API keys, and absolute filesystem paths from error messages."""
    sanitized = message
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    sanitized = _PATH_PATTERN.sub("[PATH_REDACTED]", sanitized)
    return sanitized


def _build_error_response(
    code: str,
    message: str,
    status_code: int,
    details: Optional[str] = None,
) -> JSONResponse:
    """Build a standardised JSON error response envelope."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": _sanitize_message(message),
                "details": _sanitize_message(details) if details else None,
            }
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI application."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic / FastAPI request validation errors."""
        errors = exc.errors()
        if errors:
            first_error = errors[0]
            field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
            msg = first_error.get("msg", "Validation error")
            detail_msg = f"{field}: {msg}" if field else msg
        else:
            detail_msg = "Request validation failed."

        return _build_error_response(
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            status_code=422,
            details=detail_msg,
        )

    @app.exception_handler(RAGError)
    async def rag_error_handler(
        request: Request, exc: RAGError
    ) -> JSONResponse:
        """Handle RAG orchestration errors."""
        logger.error("RAG pipeline error: %s", str(exc))
        return _build_error_response(
            code="RAG_ERROR",
            message=_sanitize_message(str(exc)),
            status_code=500,
        )

    @app.exception_handler(RerankerError)
    async def reranker_error_handler(
        request: Request, exc: RerankerError
    ) -> JSONResponse:
        """Handle reranker/retrieval errors."""
        logger.error("Retrieval error: %s", str(exc))
        return _build_error_response(
            code="RETRIEVAL_ERROR",
            message=_sanitize_message(str(exc)),
            status_code=500,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle standard HTTP exceptions (404, 405, etc.) with consistent error envelope."""
        if exc.status_code == 404:
            code = "NOT_FOUND"
        elif exc.status_code == 405:
            code = "METHOD_NOT_ALLOWED"
        else:
            code = f"HTTP_{exc.status_code}"

        return _build_error_response(
            code=code,
            message=_sanitize_message(str(exc.detail)),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unexpected exceptions. Never expose internals."""
        logger.exception("Unexpected server error: %s", str(exc))
        return _build_error_response(
            code="INTERNAL_ERROR",
            message="An unexpected internal error occurred.",
            status_code=500,
        )
