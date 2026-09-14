#!/usr/bin/env bash
# =============================================================================
# run.sh — Local Development Launcher for Lolly RAG (Bare-Metal)
#
# Launches FastAPI backend (port 8000) and Streamlit frontend (port 8501).
# Press Ctrl+C to stop both services.
#
# (For air-gapped / Docker deployments, use ./run-airgap-docker.sh instead)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load environment ─────────────────────────────────────────────────────────
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs -d '\n' 2>/dev/null) || true
fi

# ── Activate Virtual Environment ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    info "Creating virtual environment with uv..."
    uv venv --python 3.12 .venv
    uv sync
fi

source .venv/bin/activate
success "Python virtual environment active"

# ── Check Services ───────────────────────────────────────────────────────────
NEO4J_URL="${NEO4J_URL:-bolt://localhost:7687}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
    warn "Ollama is not responding at ${OLLAMA_BASE_URL}. Make sure Ollama is running ('ollama serve')."
fi

# ── Process Cleanup on Exit ──────────────────────────────────────────────────
cleanup() {
    echo ""
    info "Stopping services..."
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    wait "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    success "Services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── Start Services ───────────────────────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 ..."
(cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload) &
FASTAPI_PID=$!

info "Starting Streamlit frontend on http://localhost:8501 ..."
(cd frontend && streamlit run web_ui.py --server.address 0.0.0.0 --server.port 8501) &
STREAMLIT_PID=$!

echo ""
success "Lolly RAG local services running:"
echo "  • Streamlit UI:  http://localhost:8501"
echo "  • FastAPI docs:  http://localhost:8000/docs"
echo "  • Neo4j Browser: http://localhost:7474"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

wait "$FASTAPI_PID" "$STREAMLIT_PID"