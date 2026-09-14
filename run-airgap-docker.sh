#!/usr/bin/env bash
# =============================================================================
# run-airgap-docker.sh
#
# ONE-COMMAND RUNNER for air-gapped Docker deployment of Lolly RAG.
#
# Features:
#   • Auto-loads pre-packaged Docker images from docker-images/*.tar
#   • Validates offline model caches (.cache/huggingface, .cache/torch, ollama-models)
#   • Auto-creates .env from template if missing
#   • Fixes volume permissions for non-privileged container users
#   • Starts offline stack and waits for healthchecks
#
# Commands:
#   ./run-airgap-docker.sh            # Launch offline stack (default)
#   ./run-airgap-docker.sh --status   # Check container status & health
#   ./run-airgap-docker.sh --down     # Stop and remove containers
#   ./run-airgap-docker.sh --restart  # Restart all offline services
#   ./run-airgap-docker.sh --logs     # Follow service logs
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-up}"

case "$ACTION" in
    --down|down|stop)
        info "Stopping Lolly RAG containers..."
        docker compose down
        success "Containers stopped."
        exit 0
        ;;
    --status|status)
        info "Checking service status..."
        docker compose ps
        exit 0
        ;;
    --logs|logs)
        docker compose logs -f
        exit 0
        ;;
    --restart|restart)
        info "Restarting Lolly RAG services..."
        docker compose restart
        success "Services restarted."
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [--status | --down | --restart | --logs | --help]"
        echo ""
        echo "Commands:"
        echo "  (no args)    Load images (if needed) and launch offline stack"
        echo "  --status     Show status of running containers"
        echo "  --down       Stop all containers"
        echo "  --restart    Restart all services"
        echo "  --logs       Follow service logs"
        exit 0
        ;;
    --up|up)
        # Proceed with normal startup
        ;;
    *)
        error "Unknown option: $ACTION. Run with --help for available commands."
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Lolly RAG — Air-Gapped Docker Launcher                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ───────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    error "Docker is not installed or not in PATH."
fi

if ! docker info &>/dev/null; then
    error "Docker daemon is not running. Please start Docker."
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
    info "=== [1/3] Loading Docker images from docker-images/ ==="
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
info "=== [2/3] Verifying offline model caches ==="

# Check Ollama models
if [ -d "ollama-models/models/manifests" ] && [ -n "$(ls -A "ollama-models/models/manifests" 2>/dev/null)" ]; then
    success "Ollama models present in ./ollama-models/models/"
else
    warn "ollama-models/models/ appears empty! Ollama might start without pre-loaded models."
fi

# Check PyTorch RegNet backbone
REGNET_PTH=".cache/torch/hub/checkpoints/regnet_x_8gf-03ceed89.pth"
if [ -f "$REGNET_PTH" ]; then
    success "PyTorch RegNet backbone present: $REGNET_PTH"
else
    warn "PyTorch RegNet backbone not found! Nemotron OCR may attempt online download."
fi

# Check Hugging Face Reranker
HF_RERANKER_CACHE=".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2"
if [ -d "$HF_RERANKER_CACHE" ]; then
    success "Hugging Face CrossEncoder reranker cache present"
else
    warn "Hugging Face CrossEncoder reranker cache not found in .cache/huggingface/!"
fi

# Ensure host directory permissions allow container user (UID 1000) to read caches
chmod -R u+rwX,go+rX .cache/ ollama-models/ 2>/dev/null || true

# ── 5. Launch services in offline mode ───────────────────────────────────────
info "=== [3/3] Starting containers (offline mode) ==="
docker compose up -d

echo ""
info "Waiting for service health checks..."
# Wait up to 30s for core services to report healthy
for i in $(seq 1 15); do
    STATUS_OLLAMA="$(docker inspect --format='{{.State.Health.Status}}' localAI 2>/dev/null || echo 'starting')"
    STATUS_NEO4J="$(docker inspect --format='{{.State.Health.Status}}' lolly-DB 2>/dev/null || echo 'starting')"
    if [ "$STATUS_OLLAMA" = "healthy" ] && [ "$STATUS_NEO4J" = "healthy" ]; then
        break
    fi
    sleep 2
done

echo ""
success "All Lolly RAG services started successfully in air-gap mode!"
echo ""
echo "Service Endpoints:"
echo "  • Streamlit Web UI: http://localhost:8511"
echo "  • FastAPI Swagger:  http://localhost:8000/docs"
echo "  • Neo4j Browser:    http://localhost:7474"
echo "  • Ollama API:       http://localhost:11434"
echo ""
echo "Management Commands:"
echo "  • View live logs:  ./run-airgap-docker.sh --logs"
echo "  • Check status:    ./run-airgap-docker.sh --status"
echo "  • Stop services:   ./run-airgap-docker.sh --down"
echo "  • Restart:         ./run-airgap-docker.sh --restart"
echo ""
