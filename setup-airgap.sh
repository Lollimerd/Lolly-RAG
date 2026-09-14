#!/usr/bin/env bash
# =============================================================================
# setup-airgap.sh
#
# ONE-TIME ONLINE PREPARATION SCRIPT for Lolly RAG air-gapped deployment.
#
# Run this script on a machine WITH internet access, then copy the entire
# project directory (including the generated artifacts below) to the
# air-gapped machine.
#
# Generated artifacts:
#   docker-images/      — Pre-built & saved Docker images (FastAPI, Streamlit, Ollama, Neo4j)
#   neo4j-plugins/      — APOC and GDS JARs for Neo4j 5.26
#   ollama-models/      — Exported Ollama model manifests & blobs
#   .cache/huggingface/ — HuggingFace model cache (reranker + Nemotron OCR)
#   wheels/             — (Optional for bare-metal only) Python wheels
#
# Usage:
#   ./setup-airgap.sh                   # Docker mode (default: fast, no host wheels)
#   ./setup-airgap.sh --docker          # Explicit Docker mode
#   ./setup-airgap.sh --bare-metal      # Bare-metal mode (downloads host Python wheels)
#   ./setup-airgap.sh --all             # Prepare both Docker and bare-metal
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

# ── Mode and flag defaults ───────────────────────────────────────────────────
MODE="docker"  # default mode: docker (fast, no host wheels)
SKIP_DOCKER=false
SKIP_OLLAMA=false
SKIP_HF=false
SKIP_WHEELS=true
FORCE_ALL=false

for arg in "$@"; do
    case "$arg" in
        --docker|--docker-only)
            MODE="docker"
            SKIP_WHEELS=true
            SKIP_DOCKER=false
            ;;
        --bare-metal|--baremetal)
            MODE="bare-metal"
            SKIP_WHEELS=false
            SKIP_DOCKER=true
            ;;
        --all)
            MODE="all"
            SKIP_WHEELS=false
            SKIP_DOCKER=false
            FORCE_ALL=true
            ;;
        --skip-docker)  SKIP_DOCKER=true ;;
        --skip-ollama)  SKIP_OLLAMA=true ;;
        --skip-hf)      SKIP_HF=true ;;
        --skip-wheels)  SKIP_WHEELS=true ;;
        --help|-h)
            echo "Usage: $0 [--docker | --bare-metal | --all] [options]"
            echo ""
            echo "Modes:"
            echo "  --docker       Prepare Docker airgap artifacts (default; skips host wheels)"
            echo "  --bare-metal   Prepare bare-metal host artifacts (downloads wheels/)"
            echo "  --all          Prepare both Docker and bare-metal artifacts"
            echo ""
            echo "Granular options:"
            echo "  --skip-docker  Skip building and saving Docker images"
            echo "  --skip-ollama  Skip exporting Ollama models"
            echo "  --skip-hf      Skip downloading HuggingFace models"
            echo "  --skip-wheels  Skip downloading Python wheels"
            exit 0
            ;;
    esac
done

# ── Version pins (keep in sync with docker-compose.yml / Dockerfiles) ────────
NEO4J_VERSION="5.26"
OLLAMA_VERSION="0.32.1"
UV_VERSION="0.7.20"
PYTHON_VERSION="3.12-slim"

# Ollama models used by the app (must match init_config.py)
OLLAMA_MODELS=(
    "qwen3.5:4b"
    "qwen3.5:0.8b"
    "qwen3-embedding:0.6b"
)

# HuggingFace models
HF_RERANKER="cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_NEMOTRON="nvidia/nemotron-ocr-v2"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Lolly RAG — Simplified Air-Gap Preparation           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
info "Active mode: $MODE"
if [ "$MODE" = "docker" ]; then
    info "Docker mode: Python dependencies are packaged directly inside Docker images."
    info "(Skipping host wheel downloads to save time and disk space)."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. HuggingFace Models
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_HF" = false ]; then
    info "=== [2/4] Preparing HuggingFace models ==="
    mkdir -p .cache/huggingface/hub

    # Helper function to cache HF model (reuses local cache if already present)
    cache_hf_model() {
        local MODEL="$1"
        local SAFE_NAME="models--${MODEL//\//--}"
        local TARGET_DIR=".cache/huggingface/hub/${SAFE_NAME}"
        local HOST_CACHE="${HOME}/.cache/huggingface/hub/${SAFE_NAME}"

        if [ -d "$TARGET_DIR" ] && [ -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
            success "Model already in project cache: $MODEL"
            return 0
        fi

        if [ -d "$HOST_CACHE" ] && [ -n "$(ls -A "$HOST_CACHE" 2>/dev/null)" ]; then
            info "Copying $MODEL from existing host cache (~/.cache/huggingface)..."
            cp -rL "$HOST_CACHE" .cache/huggingface/hub/
            success "Copied: $MODEL"
            return 0
        fi

        info "Downloading $MODEL from Hugging Face Hub..."
        HF_HOME="${SCRIPT_DIR}/.cache/huggingface" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}')
"
        success "Downloaded: $MODEL"
    }

    cache_hf_model "$HF_RERANKER"
    cache_hf_model "$HF_NEMOTRON"
else
    warn "Skipping HuggingFace download (--skip-hf)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Ollama Models (Exported for direct bind mount)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_OLLAMA" = false ]; then
    info "=== [3/4] Pulling and exporting Ollama models ==="

    if ! command -v docker &>/dev/null; then
        error "docker CLI not found. Docker is required to pull Ollama models."
    fi

    OLLAMA_CONTAINER="localAI"

    if ! docker inspect "$OLLAMA_CONTAINER" &>/dev/null; then
        info "Container '$OLLAMA_CONTAINER' not found — starting via docker compose..."
        docker compose up -d local-AI
        info "Waiting for Ollama to become healthy..."
        for i in $(seq 1 30); do
            if docker exec "$OLLAMA_CONTAINER" ollama list &>/dev/null; then
                break
            fi
            sleep 2
        done
    fi

    if ! docker exec "$OLLAMA_CONTAINER" ollama list &>/dev/null; then
        error "Container '$OLLAMA_CONTAINER' is not responding. Start it with: docker compose up -d local-AI"
    fi
    success "Container '$OLLAMA_CONTAINER' is active"

    # Pull models inside container
    for MODEL in "${OLLAMA_MODELS[@]}"; do
        info "Checking/pulling model: $MODEL"
        docker exec "$OLLAMA_CONTAINER" ollama pull "$MODEL"
        success "Model ready: $MODEL"
    done

    # Export model blobs and manifests
    mkdir -p ollama-models
    info "Exporting Ollama models from container to ollama-models/models/..."
    docker cp "${OLLAMA_CONTAINER}:/root/.ollama/models" ollama-models/
    success "Ollama models saved to ollama-models/models/"
else
    warn "Skipping Ollama models (--skip-ollama)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Docker Images (Build & Export)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_DOCKER" = false ]; then
    info "=== [4/4] Building and exporting Docker images ==="

    if ! command -v docker &>/dev/null; then
        error "docker CLI not found."
    fi

    mkdir -p docker-images

    # Pull official base images
    DOCKER_BASE_IMAGES=(
        "ollama/ollama:${OLLAMA_VERSION}"
        "neo4j:${NEO4J_VERSION}"
    )

    for IMAGE in "${DOCKER_BASE_IMAGES[@]}"; do
        SAFE_NAME="${IMAGE//[:\/]/_}"
        TAR_PATH="docker-images/${SAFE_NAME}.tar"

        if [ -f "$TAR_PATH" ]; then
            success "Base image archive exists: $TAR_PATH"
        else
            info "Pulling and saving: $IMAGE"
            docker pull "$IMAGE"
            docker save "$IMAGE" -o "$TAR_PATH"
            success "Saved: $TAR_PATH ($(du -sh "$TAR_PATH" | cut -f1))"
        fi
    done

    # Build application images (uses BuildKit caching)
    info "Building application images via Docker Compose..."
    docker compose build backend-server frontend

    # Export backend image
    BACKEND_TAR="docker-images/lolly-rag-backend.tar"
    if [ ! -f "$BACKEND_TAR" ]; then
        info "Saving: lolly-rag-backend:latest → $BACKEND_TAR"
        docker save lolly-rag-backend:latest -o "$BACKEND_TAR"
        success "Saved: $BACKEND_TAR ($(du -sh "$BACKEND_TAR" | cut -f1))"
    else
        success "Backend image archive exists: $BACKEND_TAR"
    fi

    # Export frontend image
    FRONTEND_TAR="docker-images/lolly-rag-frontend.tar"
    if [ ! -f "$FRONTEND_TAR" ]; then
        info "Saving: lolly-rag-frontend:latest → $FRONTEND_TAR"
        docker save lolly-rag-frontend:latest -o "$FRONTEND_TAR"
        success "Saved: $FRONTEND_TAR ($(du -sh "$FRONTEND_TAR" | cut -f1))"
    else
        success "Frontend image archive exists: $FRONTEND_TAR"
    fi

    # Create load-images.sh helper
    cat > docker-images/load-images.sh << 'LOADSCRIPT'
#!/usr/bin/env bash
# Run this on the AIR-GAPPED machine to load all Docker images
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Loading Docker images from $SCRIPT_DIR ..."
for TAR in "$SCRIPT_DIR"/*.tar; do
    echo "  Loading: $(basename "$TAR")"
    docker load -i "$TAR"
done
echo "All Docker images loaded successfully. Verify with: docker images"
LOADSCRIPT
    chmod +x docker-images/load-images.sh
    success "Generated docker-images/load-images.sh"
else
    warn "Skipping Docker image export (--skip-docker)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Optional: Python Wheels (Bare-metal only)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_WHEELS" = false ]; then
    info "=== [Optional] Downloading Python wheels for bare-metal ==="
    mkdir -p wheels/backend wheels/frontend

    info "Downloading backend wheels..."
    pip download -r backend/requirements.txt --dest wheels/backend
    success "Backend wheels saved to wheels/backend/"

    info "Downloading frontend wheels..."
    pip download -r frontend/requirements.txt --dest wheels/frontend
    success "Frontend wheels saved to wheels/frontend/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PREPARATION COMPLETE — TRANSFER CHECKLIST      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Copy the entire project directory to the air-gapped machine."
echo ""
echo "  Generated Artifacts:"
[ "$SKIP_HF" = false ]     && echo "    ✅ .cache/huggingface/      — HuggingFace reranker & OCR models"
[ "$SKIP_OLLAMA" = false ] && echo "    ✅ ollama-models/models/    — Ollama model manifests & blobs"
[ "$SKIP_DOCKER" = false ] && echo "    ✅ docker-images/           — Docker image tar archives"
[ "$SKIP_WHEELS" = false ] && echo "    ✅ wheels/                  — Python wheels (bare-metal only)"
echo ""
echo "  To launch on the AIR-GAPPED machine (Docker):"
echo "    chmod +x run-airgap-docker.sh"
echo "    ./run-airgap-docker.sh"
echo ""
