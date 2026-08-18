"""Phase 8: FastAPI Production Application.

Creates and configures the FastAPI application with:
- Lifespan-managed service initialisation (warm models at startup).
- CORS with explicit configuration.
- Security-oriented response headers.
- Structured logging.
- Custom exception handlers.
- API router registration.

Start locally with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.api.dependencies import initialize_services
from app.api.errors import register_exception_handlers
from app.api.routes import router
from app.config import API_CORS_ORIGINS, API_TITLE, API_VERSION
from app.observability.middleware import ObservabilityMiddleware

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add basic security-related response headers to every response."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[arg-type]
    """Manage application lifecycle: warm services on startup, cleanup on shutdown."""
    logger.info("Starting Nexora RAG API — initialising services...")
    try:
        initialize_services()
        logger.info("Service initialisation complete.")
    except Exception as exc:
        logger.warning("Service initialisation encountered issues: %s", exc)
    yield
    logger.info("Shutting down Nexora RAG API.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Production RAG API exposing grounded retrieval-augmented generation "
            "with hybrid search, cross-encoder reranking, and citation verification."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — explicit configuration, not unrestricted wildcard
    application.add_middleware(
        CORSMiddleware,
        allow_origins=API_CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    # Observability & request correlation
    application.add_middleware(ObservabilityMiddleware)

    # Security headers
    application.add_middleware(SecurityHeadersMiddleware)

    # Custom exception handlers
    register_exception_handlers(application)

    # API routes
    application.include_router(router)

    return application


# Module-level application instance for uvicorn
app = create_app()
