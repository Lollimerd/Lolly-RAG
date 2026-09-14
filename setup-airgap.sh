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
#   ollama-models/      — Exported Ollama model manifests & blobs
#   .cache/huggingface/ — HuggingFace model cache (verified offline reranker + Nemotron OCR)
#   .cache/torch/       — PyTorch Hub checkpoints (Nemotron RegNet backbone)
#   wheels/             — (Optional for bare-metal only) Python wheels
#
# Usage:
#   ./setup-airgap.sh                   # Docker mode (default: fast, idempotent)
#   ./setup-airgap.sh --force           # Force re-download/re-build all artifacts
#   ./setup-airgap.sh --verify          # Verify all local artifacts offline without downloading
#   ./setup-airgap.sh --docker          # Explicit Docker mode
#   ./setup-airgap.sh --bare-metal      # Bare-metal mode (downloads host Python wheels)
#   ./setup-airgap.sh --all             # Prepare both Docker and bare-metal
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

# ── Mode and flag defaults ───────────────────────────────────────────────────
MODE="docker"
SKIP_DOCKER=false
SKIP_OLLAMA=false
SKIP_HF=false
SKIP_WHEELS=true
FORCE_ALL=false
VERIFY_ONLY=false

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
            ;;
        --force|-f)
            FORCE_ALL=true
            ;;
        --verify)
            VERIFY_ONLY=true
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
            echo "Options:"
            echo "  --force, -f    Force re-download / re-build all components even if cached"
            echo "  --verify       Verify existing airgap artifacts offline without downloading"
            echo "  --skip-docker  Skip building and saving Docker images"
            echo "  --skip-ollama  Skip pulling/exporting Ollama models"
            echo "  --skip-hf      Skip downloading HuggingFace and PyTorch models"
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
echo "║        Lolly RAG — Optimized Air-Gap Preparation            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
info "Active mode: $MODE (Force: $FORCE_ALL, Verify only: $VERIFY_ONLY)"
if [ "$MODE" = "docker" ]; then
    info "Docker mode: Python dependencies are packaged directly inside Docker images."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1. HuggingFace Models & PyTorch Checkpoints (Strict Verification)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_HF" = false ]; then
    info "=== [1/3] Preparing & Verifying HuggingFace and PyTorch Models ==="
    mkdir -p .cache/huggingface/hub .cache/torch/hub/checkpoints

    # 1.1 PyTorch RegNet backbone for Nemotron OCR
    REGNET_PTH="regnet_x_8gf-03ceed89.pth"
    REGNET_URL="https://download.pytorch.org/models/${REGNET_PTH}"
    TARGET_REGNET=".cache/torch/hub/checkpoints/${REGNET_PTH}"
    HOST_REGNET="${HOME}/.cache/torch/hub/checkpoints/${REGNET_PTH}"

    # Minimum size check (RegNet X 8GF is ~150MB)
    MIN_REGNET_BYTES=100000000

    if [ -f "$TARGET_REGNET" ] && [ "$(stat -c%s "$TARGET_REGNET" 2>/dev/null || stat -f%z "$TARGET_REGNET")" -gt "$MIN_REGNET_BYTES" ] && [ "$FORCE_ALL" = false ]; then
        success "PyTorch RegNet backbone verified: $TARGET_REGNET ($(du -sh "$TARGET_REGNET" | cut -f1))"
    elif [ "$VERIFY_ONLY" = true ]; then
        error "Verification failed: $TARGET_REGNET missing or incomplete."
    elif [ -f "$HOST_REGNET" ] && [ "$(stat -c%s "$HOST_REGNET" 2>/dev/null || stat -f%z "$HOST_REGNET")" -gt "$MIN_REGNET_BYTES" ]; then
        info "Copying $REGNET_PTH from host ~/.cache/torch..."
        cp -L "$HOST_REGNET" "$TARGET_REGNET"
        success "Copied & verified: $REGNET_PTH"
    else
        info "Downloading $REGNET_PTH from PyTorch Hub..."
        curl -fsSL -o "$TARGET_REGNET" "$REGNET_URL"
        success "Downloaded: $REGNET_PTH ($(du -sh "$TARGET_REGNET" | cut -f1))"
    fi

    # 1.2 HuggingFace Reranker and Nemotron Models with Snapshot Integrity Check
    verify_or_download_hf_model() {
        local MODEL="$1"
        local FORCE_FLAG="$2"
        local VERIFY_FLAG="$3"
        local SAFE_NAME="models--${MODEL//\//--}"
        local HOST_CACHE="${HOME}/.cache/huggingface/hub/${SAFE_NAME}"

        info "Processing Hugging Face model: $MODEL ..."

        HF_HOME="${SCRIPT_DIR}/.cache/huggingface" python3 - <<EOF
import os, sys, shutil
from pathlib import Path

model_id = "${MODEL}"
force = ("${FORCE_FLAG}" == "true")
verify_only = ("${VERIFY_FLAG}" == "true")

hf_home = Path("${SCRIPT_DIR}/.cache/huggingface")
hub_dir = hf_home / "hub" / "${SAFE_NAME}"
snapshots_dir = hub_dir / "snapshots"
refs_main = hub_dir / "refs" / "main"
host_cache = Path("${HOST_CACHE}")

def is_snapshot_complete(snap_path: Path, model_id: str) -> bool:
    if not snap_path.is_dir():
        return False
    if not (snap_path / "config.json").exists():
        return False
    # Check for weights
    has_weights = (
        (snap_path / "model.safetensors").exists()
        or (snap_path / "pytorch_model.bin").exists()
        or any(snap_path.glob("*.safetensors"))
        or any(snap_path.glob("*.bin"))
        or (snap_path / "v2_english").exists()
        or (snap_path / "nemotron-ocr").exists()
    )
    return has_weights

def fix_and_verify(model_id: str) -> bool:
    if not snapshots_dir.is_dir():
        return False
    valid_snaps = [s for s in snapshots_dir.iterdir() if is_snapshot_complete(s, model_id)]
    if not valid_snaps:
        return False

    best_snap = valid_snaps[0]
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    current_ref = refs_main.read_text().strip() if refs_main.exists() else None

    # Ensure refs/main points to a complete snapshot
    if current_ref != best_snap.name:
        print(f"  [FIX] Correcting refs/main pointer: -> {best_snap.name}")
        refs_main.write_text(best_snap.name)

    # Sync missing files between snapshots so that ANY requested revision works offline
    for snap in snapshots_dir.iterdir():
        if snap.is_dir() and snap != best_snap:
            for item in best_snap.iterdir():
                dest = snap / item.name
                if not dest.exists():
                    try:
                        dest.symlink_to(item)
                    except Exception:
                        if item.is_file():
                            shutil.copy2(item, dest)
                        elif item.is_dir():
                            shutil.copytree(item, dest)

    # Validate offline loading
    try:
        if "cross-encoder" in model_id:
            from transformers import AutoConfig, AutoTokenizer
            AutoConfig.from_pretrained(model_id, local_files_only=True)
            AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        elif "nemotron" in model_id:
            if not ((best_snap / "v2_english").exists() or (best_snap / "nemotron-ocr").exists()):
                return False
        return True
    except Exception as exc:
        print(f"  [WARN] Offline test check raised: {exc}")
        return False

# Step 1: Check existing project cache
if not force and fix_and_verify(model_id):
    print(f"  [OK] Snapshot verified and offline-ready: {model_id}")
    sys.exit(0)

if verify_only:
    print(f"  [ERROR] Model snapshot verification failed for: {model_id}")
    sys.exit(1)

# Step 2: Try importing from host cache if available
if not force and host_cache.is_dir():
    print(f"  [INFO] Importing from host cache ({host_cache})...")
    shutil.copytree(host_cache, hub_dir, dirs_exist_ok=True, symlinks=True)
    if fix_and_verify(model_id):
        print(f"  [OK] Imported and verified from host cache: {model_id}")
        sys.exit(0)

# Step 3: Download complete snapshot from Hugging Face Hub
print(f"  [INFO] Downloading complete snapshot from Hugging Face Hub: {model_id}...")
from huggingface_hub import snapshot_download
ignore = ["*.onnx", "*.onnx_data", "onnx/*"] if "cross-encoder" in model_id else None
snapshot_download(repo_id=model_id, local_files_only=False, ignore_patterns=ignore)

# Step 4: Final post-download verification
if fix_and_verify(model_id):
    print(f"  [OK] Successfully downloaded and verified: {model_id}")
else:
    print(f"  [ERROR] Snapshot verification failed after download for: {model_id}")
    sys.exit(1)
EOF
    }

    verify_or_download_hf_model "$HF_RERANKER" "$FORCE_ALL" "$VERIFY_ONLY"
    verify_or_download_hf_model "$HF_NEMOTRON" "$FORCE_ALL" "$VERIFY_ONLY"
else
    warn "Skipping HuggingFace and PyTorch models (--skip-hf)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Ollama Models (Safe Pull & Bind-Mount Verification)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_OLLAMA" = false ]; then
    info "=== [2/3] Preparing Ollama Models ==="

    if ! command -v docker &>/dev/null; then
        error "docker CLI not found. Docker is required to manage Ollama models."
    fi

    mkdir -p ollama-models/models

    if [ "$VERIFY_ONLY" = true ]; then
        if [ -d "ollama-models/models/manifests" ] && [ -n "$(ls -A "ollama-models/models/manifests" 2>/dev/null)" ]; then
            success "Ollama models directory verified: ./ollama-models/models/ ($(du -sh ollama-models/models | cut -f1))"
        else
            error "Verification failed: ollama-models/models/manifests is missing or empty."
        fi
    else
        OLLAMA_CONTAINER="localAI"

        # Ensure container is up
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

        # Check existing models to avoid redundant multi-gigabyte re-downloads
        INSTALLED_MODELS="$(docker exec "$OLLAMA_CONTAINER" ollama list 2>/dev/null || true)"

        for MODEL in "${OLLAMA_MODELS[@]}"; do
            if [ "$FORCE_ALL" = false ] && echo "$INSTALLED_MODELS" | grep -E -q "^${MODEL}[[:space:]]"; then
                success "Ollama model already cached: $MODEL"
            else
                info "Pulling model: $MODEL (this may take a few moments)..."
                docker exec "$OLLAMA_CONTAINER" ollama pull "$MODEL"
                success "Model ready: $MODEL"
            fi
        done

        # Verify host bind-mount sync (docker-compose binds ./ollama-models/models -> /root/.ollama/models)
        if [ -d "ollama-models/models/manifests" ] && [ -n "$(ls -A "ollama-models/models/manifests" 2>/dev/null)" ]; then
            success "Ollama models already synced to host via bind mount: ./ollama-models/models/ ($(du -sh ollama-models/models | cut -f1))"
        else
            info "Exporting Ollama models from container to ./ollama-models/models/..."
            docker cp "${OLLAMA_CONTAINER}:/root/.ollama/models/." ollama-models/models/ 2>/dev/null || true
            success "Ollama models exported to ./ollama-models/models/"
        fi
    fi
else
    warn "Skipping Ollama models (--skip-ollama)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Docker Images (Build & Export)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_DOCKER" = false ]; then
    info "=== [3/3] Building & Exporting Docker Images ==="

    if ! command -v docker &>/dev/null; then
        error "docker CLI not found."
    fi

    mkdir -p docker-images

    # 3.1 Base Images
    DOCKER_BASE_IMAGES=(
        "ollama/ollama:${OLLAMA_VERSION}"
        "neo4j:${NEO4J_VERSION}"
    )

    for IMAGE in "${DOCKER_BASE_IMAGES[@]}"; do
        SAFE_NAME="${IMAGE//[:\/]/_}"
        TAR_PATH="docker-images/${SAFE_NAME}.tar"

        if [ -f "$TAR_PATH" ] && [ "$FORCE_ALL" = false ]; then
            success "Base image archive exists: $TAR_PATH ($(du -sh "$TAR_PATH" | cut -f1))"
        elif [ "$VERIFY_ONLY" = true ]; then
            error "Verification failed: Image tarball missing: $TAR_PATH"
        else
            info "Pulling and saving: $IMAGE"
            docker pull "$IMAGE"
            docker save "$IMAGE" -o "$TAR_PATH"
            success "Saved: $TAR_PATH ($(du -sh "$TAR_PATH" | cut -f1))"
        fi
    done

    # 3.2 Application Images (FastAPI Backend & Streamlit Frontend)
    BACKEND_TAR="docker-images/lolly-rag-backend.tar"
    FRONTEND_TAR="docker-images/lolly-rag-frontend.tar"

    if [ "$VERIFY_ONLY" = true ]; then
        [ -f "$BACKEND_TAR" ] || error "Verification failed: $BACKEND_TAR missing"
        [ -f "$FRONTEND_TAR" ] || error "Verification failed: $FRONTEND_TAR missing"
        success "Application image archives verified: $BACKEND_TAR, $FRONTEND_TAR"
    else
        info "Building application images via Docker Compose..."
        docker compose build backend-server frontend

        if [ ! -f "$BACKEND_TAR" ] || [ "$FORCE_ALL" = true ]; then
            info "Saving: lolly-rag-backend:latest → $BACKEND_TAR"
            docker save lolly-rag-backend:latest -o "$BACKEND_TAR"
            success "Saved: $BACKEND_TAR ($(du -sh "$BACKEND_TAR" | cut -f1))"
        else
            success "Backend image archive exists: $BACKEND_TAR ($(du -sh "$BACKEND_TAR" | cut -f1))"
        fi

        if [ ! -f "$FRONTEND_TAR" ] || [ "$FORCE_ALL" = true ]; then
            info "Saving: lolly-rag-frontend:latest → $FRONTEND_TAR"
            docker save lolly-rag-frontend:latest -o "$FRONTEND_TAR"
            success "Saved: $FRONTEND_TAR ($(du -sh "$FRONTEND_TAR" | cut -f1))"
        else
            success "Frontend image archive exists: $FRONTEND_TAR ($(du -sh "$FRONTEND_TAR" | cut -f1))"
        fi
    fi

    # 3.3 Generate load-images.sh helper
    cat > docker-images/load-images.sh << 'LOADSCRIPT'
#!/usr/bin/env bash
# =============================================================================
# Run this on the AIR-GAPPED machine to load all Docker images into local daemon
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Loading Docker images from $SCRIPT_DIR ..."
for TAR in "$SCRIPT_DIR"/*.tar; do
    [ -f "$TAR" ] || continue
    echo "  Loading: $(basename "$TAR") ..."
    docker load -i "$TAR"
done
echo "All Docker images loaded successfully. Verify with: docker images"
LOADSCRIPT
    chmod +x docker-images/load-images.sh
    success "Generated docker-images/load-images.sh"
else
    warn "Skipping Docker images (--skip-docker)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Optional: Python Wheels (Bare-metal only)
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
# Summary & Transfer Checklist
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PREPARATION COMPLETE — AIR-GAP READY           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Artifact Status:"
[ -d ".cache/huggingface" ] && echo "    ✅ .cache/huggingface/      ($(du -sh .cache/huggingface | cut -f1)) — Reranker & OCR models"
[ -d ".cache/torch" ]       && echo "    ✅ .cache/torch/            ($(du -sh .cache/torch | cut -f1)) — PyTorch RegNet backbone"
[ -d "ollama-models" ]      && echo "    ✅ ollama-models/           ($(du -sh ollama-models | cut -f1)) — Ollama model blobs & manifests"
[ -d "docker-images" ]      && echo "    ✅ docker-images/           ($(du -sh docker-images | cut -f1)) — Docker images (.tar)"
[ "$SKIP_WHEELS" = false ]  && echo "    ✅ wheels/                  ($(du -sh wheels | cut -f1)) — Python wheels"
echo ""
echo "  Transfer Instructions:"
echo "    1. Copy this entire project directory to the air-gapped machine."
echo "    2. On the air-gapped machine, simply run:"
echo "         ./run-airgap-docker.sh"
echo ""
