FROM python:3.11-slim AS builder
WORKDIR /app

# Set environment variables for builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies if any (e.g., for C extensions)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*
# For now, assuming no special build-time system deps are needed for these packages

COPY requirements.txt .

# Create wheels
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

FROM python:3.11-slim AS prod
WORKDIR /app

# Set environment variables for runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Create non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app \
    && chown -R appuser:appuser /app

# Copy wheels from builder stage
COPY --from=builder /wheels /wheels

# Install dependencies from wheels
# Ensure requirements.txt is also available to guide the installation from wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Copy application code
# Ensure .dockerignore is properly configured to avoid copying unnecessary files
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}

# --- Development Stage ---
FROM python:3.11-slim AS dev
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies if needed for dev tools (e.g. git)
# RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-dev.txt .

# Install all dependencies for development (requirements-dev.txt should include -r requirements.txt)
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# Expose the port the app runs on, if running the app in dev
EXPOSE 8000
# Default command for dev (e.g., run the app, or could be bash)
CMD ["python", "main.py"]
