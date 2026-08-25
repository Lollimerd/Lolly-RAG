# syntax=docker/dockerfile:1

FROM python:3.13-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy \
    HF_HOME="/home/appuser/.cache/huggingface"

WORKDIR /app

# Copy uv binary for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Install dependencies into system Python environment using uv cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/requirements.txt,target=requirements.txt \
    uv pip install --system -r requirements.txt

# Download reranker model from HuggingFace
RUN hf download cross-encoder/ms-marco-MiniLM-L-6-v2

# Create cache directories for HuggingFace and models, ensure appuser owns them so it can write models/logs
RUN mkdir -p /home/appuser/.cache/huggingface /app/backend \
    && chown -R appuser:appuser /home/appuser /app

# Copy relevant backend source code
# --chown ensures files are owned by appuser immediately
COPY --chown=appuser:appuser backend/ /app/backend/

# Switch to non-privileged user
USER appuser

EXPOSE 8000

# Set working directory to backend folder
WORKDIR /app/backend

# Run the FastAPI app
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]