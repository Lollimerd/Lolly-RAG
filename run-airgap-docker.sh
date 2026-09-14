#!/usr/bin/env bash
# =============================================================================
# run-airgap-docker.sh
#
# ONE-COMMAND RUNNER for air-gapped Docker deployment of Lolly RAG.
#
# What this script does:
#   1. Loads pre-packaged Docker images from docker-images/*.tar (if needed)
#   2. Ensures .env exists (copies from .env.example if missing)
#   3. Launches the complete offline stack using docker-compose.yml
#
# Usage:
#   chmod +x run-airgap-docker.sh
#   ./run-airgap-docker.sh
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Lolly RAG — Air-Gapped Docker Launcher                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH."
fi

# ── 2. Load Docker images if not already loaded ──────────────────────────────
REQUIRED_IMAGES=(
    "ollama/ollama:0.32.1"
    "neo4j:5.26"
    "lolly-rag-backend:latest"
    "lolly-rag-frontend:latest"
)

LOAD_NEEDED=false
for IMG in "${REQUIRED_IMAGES[@]}"; do
    if ! docker image inspect "$IMG" &>/dev/null; then
        LOAD_NEEDED=true
        break
    fi
done

if [ "$LOAD_NEEDED" = true ]; then
    info "=== [1/2] Loading Docker images from docker-images/ ==="
    if [ -d "docker-images" ] && [ -f "docker-images/load-images.sh" ]; then
        bash docker-images/load-images.sh
    elif [ -d "docker-images" ]; then
        for TAR in docker-images/*.tar; do
            [ -f "$TAR" ] || continue
            info "Loading $TAR ..."
            docker load -i "$TAR"
        done
    else
        warn "docker-images/ directory not found. Assuming images are already present."
    fi
    success "Docker images ready"
else
    success "All required Docker images are already loaded in local daemon"
fi

# ── 3. Check environment configuration ───────────────────────────────────────
if [ ! -f ".env" ]; then
    info "No .env file found — copying default template from .env.example..."
    cp .env.example .env
    success "Created .env"
fi

# ── 4. Verify model and plugin artifacts ─────────────────────────────────────
if [ ! -d "ollama-models/models" ]; then
    warn "ollama-models/models/ directory not found! Ollama may start without pre-loaded models."
fi


# ── 5. Launch services in offline mode ───────────────────────────────────────
info "=== [2/2] Starting containers (offline mode) ==="
docker compose up -d

echo ""
success "All Lolly RAG services started successfully in air-gap mode!"
echo ""
echo "Service Endpoints:"
echo "  • Streamlit Web UI: http://localhost:8511"
echo "  • FastAPI Swagger:  http://localhost:8000/docs"
echo "  • Neo4j Browser:    http://localhost:7474"
echo "  • Ollama API:       http://localhost:11434"
echo ""
echo "Logs can be viewed with:"
echo "  docker compose logs -f"
echo ""
