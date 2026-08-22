# syntax=docker/dockerfile:1

FROM python:3.13-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Copy uv binary for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Install dependencies into system Python environment using uv cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv pip install --system -r requirements.txt

# Prepare Streamlit cache and config directories, ensure appuser owns them
RUN mkdir -p /home/appuser/.cache /home/appuser/.streamlit /app/frontend \
    && chown -R appuser:appuser /home/appuser /app

# Copy relevant frontend source code
# --chown ensures files are owned by appuser immediately
COPY --chown=appuser:appuser frontend/ /app/frontend/

# Switch to non-privileged user
USER appuser

EXPOSE 8510

# Set working directory to frontend folder
WORKDIR /app/frontend

# Run Streamlit
CMD ["streamlit", "run", "web.py", "--server.address=0.0.0.0"]