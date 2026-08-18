# ============================================================
# Nexora RAG API — Production Dockerfile
# Phase 8: CPU-oriented, minimal Python 3.11 image
# ============================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Install system dependencies
# ---------------------------------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy application code and data
# ---------------------------------------------------------------------------
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# ---------------------------------------------------------------------------
# Security: Create non-root user and set permissions
# ---------------------------------------------------------------------------
RUN useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# ---------------------------------------------------------------------------
# Expose API port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Health check (Liveness probe)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---------------------------------------------------------------------------
# Start Uvicorn
# ---------------------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
