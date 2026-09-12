# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/backend:/app" \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy \
    HF_HOME="/home/appuser/.cache/huggingface"

WORKDIR /app

# Copy uv binary for fast, reliable dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-privileged user matching host UID for shared volume permissions
ARG UID=1000
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Install runtime system libraries for Nemotron OCR, image processing, and model loading
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into system Python environment using uv cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system -r /tmp/requirements.txt

# Create cache directories and set permissions
RUN mkdir -p /home/appuser/.cache/huggingface /app/backend \
    && chown -R appuser:appuser /home/appuser /app

# Copy backend source code with appuser ownership
COPY --chown=appuser:appuser backend/ /app/backend/

# Switch to non-privileged user
USER appuser

EXPOSE 8000
WORKDIR /app/backend

# Run the FastAPI app
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]