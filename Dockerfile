# syntax=docker/dockerfile:1.4

# ─── Builder stage ───────────────────────────────────
FROM python:3.11-slim AS builder

# Install build deps with cache mounts for speed and reliability
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependencies to leverage layer cache
COPY requirements-core.txt requirements-rag.txt requirements-dev.txt ./

# Use pip cache mount for faster wheel builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel && \
    pip wheel --no-cache-dir --wheel-dir wheels \
      -r requirements-core.txt \
      -r requirements-rag.txt \
      -r requirements-dev.txt


# ─── Runtime stage ───────────────────────────────────
FROM python:3.11-slim AS runtime

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install wheels with no-index, then clean up wheels
COPY --from=builder /build/wheels /wheels
COPY requirements-core.txt requirements-rag.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-index --find-links=/wheels \
      -r requirements-core.txt -r requirements-rag.txt && \
    rm -rf /wheels requirements-*.txt

# Copy source code and set correct ownership in one layer
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check to ensure container is healthy
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default production command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# ─── Development stage ───────────────────────────────
FROM runtime AS development

# Switch back to root to install dev dependencies
USER root

# Install development dependencies
COPY --from=builder /build/wheels /wheels
COPY requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-index --find-links=/wheels \
      -r requirements-dev.txt && \
    rm -rf /wheels requirements-dev.txt

# Switch back to non-root user
USER appuser

# Command for development with hot-reload
# Note: Mount your code as a volume with: -v "$(pwd):/app"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
