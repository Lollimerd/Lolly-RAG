ARG UV_VERSION=0.7.20

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy

WORKDIR /app

# Copy uv binary from the pinned intermediate stage
COPY --from=uv-bin /uv /uvx /bin/

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Install Python dependencies
COPY frontend/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.txt

# Prepare Streamlit cache and config directories, ensure appuser owns them
RUN mkdir -p /home/appuser/.cache /home/appuser/.streamlit /app/frontend \
    && chown -R appuser:appuser /home/appuser /app

# Copy relevant frontend source code
COPY --chown=appuser:appuser frontend/ /app/frontend/

# Switch to non-privileged user
USER appuser

EXPOSE 8501
WORKDIR /app/frontend

# Run Streamlit
CMD ["streamlit", "run", "web_ui.py", "--server.address=0.0.0.0"]