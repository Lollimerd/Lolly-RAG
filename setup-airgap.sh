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
#   .venv/              — (For bare-metal) Python 3.12 virtual environment (synced via uv)
#
# Usage:
#   ./setup-airgap.sh                   # Docker mode (default: fast, idempotent)
#   ./setup-airgap.sh --force           # Force re-download/re-build all artifacts
#   ./setup-airgap.sh --verify          # Verify all local artifacts offline without downloading
#   ./setup-airgap.sh --docker          # Explicit Docker mode
#   ./setup-airgap.sh --bare-metal      # Bare-metal mode (prepares .venv using uv sync)
#   ./setup-airgap.sh --all             # Prepare both Docker and bare-metal
# =============================================================================

set -euo pipefail
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# =============================================================================
# CONFIGURATION — keep in sync with docker-compose.yml / Dockerfiles
# =============================================================================

readonly NEO4J_VERSION="5.26"
readonly OLLAMA_VERSION="0.33.2"

# Ollama models used by the app (must match init_config.py)
readonly OLLAMA_MODELS=(
    "qwen3.5:4b"
    "qwen3.5:0.8b"
    "qwen3-embedding:0.6b"
)

# HuggingFace models
readonly HF_RERANKER="cross-encoder/ms-marco-MiniLM-L-6-v2"
readonly HF_NEMOTRON="nvidia/nemotron-ocr-v2"

# Minimum file-size sanity check for RegNet checkpoint (~150 MB)
readonly MIN_REGNET_BYTES=100000000
readonly REGNET_FILENAME="regnet_x_8gf-03ceed89.pth"
readonly REGNET_URL="https://download.pytorch.org/models/${REGNET_FILENAME}"

# =============================================================================
# COLOUR HELPERS
# =============================================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# =============================================================================
# CLI PARSING
# =============================================================================

MODE="docker"
SKIP_DOCKER=false
SKIP_OLLAMA=false
SKIP_HF=false
SKIP_BARE_METAL=true
FORCE_ALL=false
VERIFY_ONLY=false

parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --docker|--docker-only)
                MODE="docker"; SKIP_BARE_METAL=true; SKIP_DOCKER=false ;;
            --bare-metal|--baremetal)
                MODE="bare-metal"; SKIP_BARE_METAL=false; SKIP_DOCKER=true ;;
            --all)
                MODE="all"; SKIP_BARE_METAL=false; SKIP_DOCKER=false ;;
            --force|-f)
                FORCE_ALL=true ;;
            --verify)
                VERIFY_ONLY=true ;;
            --skip-docker)  SKIP_DOCKER=true ;;
            --skip-ollama)  SKIP_OLLAMA=true ;;
            --skip-hf)      SKIP_HF=true ;;
            --skip-bare-metal|--skip-wheels|--skip-venv)  SKIP_BARE_METAL=true ;;
            --help|-h)
                print_help; exit 0 ;;
        esac
    done
}

print_help() {
    echo "Usage: $0 [--docker | --bare-metal | --all] [options]"
    echo ""
    echo "Modes:"
    echo "  --docker          Prepare Docker airgap artifacts (default; skips bare-metal .venv)"
    echo "  --bare-metal      Prepare bare-metal host artifacts (syncs .venv via uv)"
    echo "  --all             Prepare both Docker and bare-metal artifacts"
    echo ""
    echo "Options:"
    echo "  --force, -f       Force re-download / re-build all components even if cached"
    echo "  --verify          Verify existing airgap artifacts offline without downloading"
    echo "  --skip-docker     Skip building and saving Docker images"
    echo "  --skip-ollama     Skip pulling/exporting Ollama models"
    echo "  --skip-hf         Skip downloading HuggingFace and PyTorch models"
    echo "  --skip-bare-metal Skip preparing bare-metal .venv (alias: --skip-wheels)"
}

# =============================================================================
# PHASE 1 — HuggingFace Models & PyTorch Checkpoints
# =============================================================================

# Returns 0 if the file exists and is large enough, 1 otherwise.
_regnet_is_valid() {
    local path="$1"
    [ -f "$path" ] \
        && [ "$(stat -c%s "$path" 2>/dev/null || stat -f%z "$path")" -gt "$MIN_REGNET_BYTES" ]
}

prepare_pytorch_regnet() {
    local target=".cache/torch/hub/checkpoints/${REGNET_FILENAME}"
    local host_copy="${HOME}/.cache/torch/hub/checkpoints/${REGNET_FILENAME}"

    if [ "$VERIFY_ONLY" = true ]; then
        if _regnet_is_valid "$target"; then
            success "PyTorch RegNet backbone verified: $target ($(du -sh "$target" | cut -f1))"
            return
        else
            error "Verification failed: $target missing or too small."
        fi
    fi

    if _regnet_is_valid "$target" && [ "$FORCE_ALL" = false ]; then
        success "PyTorch RegNet backbone verified: $target ($(du -sh "$target" | cut -f1))"
        return
    fi

    if _regnet_is_valid "$host_copy"; then
        info "Copying $REGNET_FILENAME from host ~/.cache/torch ..."
        cp -L "$host_copy" "$target"
        success "Copied & verified: $REGNET_FILENAME"
    else
        info "Downloading $REGNET_FILENAME from PyTorch Hub ..."
        curl -fsSL -o "$target" "$REGNET_URL"
        success "Downloaded: $REGNET_FILENAME ($(du -sh "$target" | cut -f1))"
    fi
}

# Verifies (and optionally downloads) a single HuggingFace model snapshot.
prepare_hf_model() {
    local model="$1"
    local safe_name="models--${model//\//--}"
    local host_cache="${HOME}/.cache/huggingface/hub/${safe_name}"

    info "Processing Hugging Face model: $model ..."

    HF_HOME="${SCRIPT_DIR}/.cache/huggingface" \
    "${SCRIPT_DIR}/.venv/bin/python3" - <<EOF
import os, sys, shutil
from pathlib import Path

model_id    = "${model}"
force       = ("${FORCE_ALL}"  == "true")
verify_only = ("${VERIFY_ONLY}" == "true")

hf_home      = Path("${SCRIPT_DIR}/.cache/huggingface")
hub_dir      = hf_home / "hub" / "${safe_name}"
snapshots_dir = hub_dir / "snapshots"
refs_main    = hub_dir / "refs" / "main"
host_cache   = Path("${host_cache}")

# ── Helpers ────────────────────────────────────────────────────────────────

def snapshot_is_complete(snap: Path, mid: str) -> bool:
    """True when a snapshot directory contains a config + at least one weight file."""
    if not snap.is_dir() or not (snap / "config.json").exists():
        return False
    return (
        (snap / "model.safetensors").exists()
        or (snap / "pytorch_model.bin").exists()
        or any(snap.glob("*.safetensors"))
        or any(snap.glob("*.bin"))
        or (snap / "v2_english").exists()     # Nemotron OCR layout
        or (snap / "nemotron-ocr").exists()
    )

def heal_and_verify(mid: str) -> bool:
    """
    Ensure refs/main points at a complete snapshot, back-fill symlinks
    across all snapshot revisions, then validate offline loading.
    """
    if not snapshots_dir.is_dir():
        return False

    valid = [s for s in snapshots_dir.iterdir() if snapshot_is_complete(s, mid)]
    if not valid:
        return False

    best = valid[0]
    refs_main.parent.mkdir(parents=True, exist_ok=True)

    # Fix stale refs/main pointer
    current_ref = refs_main.read_text().strip() if refs_main.exists() else None
    if current_ref != best.name:
        print(f"  [FIX] refs/main -> {best.name}")
        refs_main.write_text(best.name)

    # Back-fill missing files in older snapshot revisions
    for snap in snapshots_dir.iterdir():
        if snap.is_dir() and snap != best:
            for item in best.iterdir():
                dest = snap / item.name
                if not dest.exists():
                    try:
                        dest.symlink_to(item)
                    except Exception:
                        if item.is_file():
                            shutil.copy2(item, dest)
                        elif item.is_dir():
                            shutil.copytree(item, dest)

    # Validate that transformers can load the model without network access
    try:
        if "cross-encoder" in mid:
            from transformers import AutoConfig, AutoTokenizer
            AutoConfig.from_pretrained(mid, local_files_only=True)
            AutoTokenizer.from_pretrained(mid, local_files_only=True)
        elif "nemotron" in mid:
            if not ((best / "v2_english").exists() or (best / "nemotron-ocr").exists()):
                return False
        return True
    except Exception as exc:
        print(f"  [WARN] Offline validation: {exc}")
        return False

# ── Resolution order ───────────────────────────────────────────────────────

if verify_only:
    if heal_and_verify(model_id):
        print(f"  [OK] Snapshot verified and offline-ready: {model_id}")
        sys.exit(0)
    else:
        print(f"  [ERROR] Snapshot verification failed: {model_id}")
        sys.exit(1)

# 1. Already cached in project directory
if not force and heal_and_verify(model_id):
    print(f"  [OK] Snapshot verified and offline-ready: {model_id}")
    sys.exit(0)

# 2. Import from host's ~/.cache/huggingface
if not force and host_cache.is_dir():
    print(f"  [INFO] Importing from host cache ({host_cache}) ...")
    shutil.copytree(host_cache, hub_dir, dirs_exist_ok=True, symlinks=True)
    if heal_and_verify(model_id):
        print(f"  [OK] Imported and verified from host cache: {model_id}")
        sys.exit(0)

# 3. Download fresh snapshot from Hugging Face Hub
print(f"  [INFO] Downloading snapshot from Hugging Face Hub: {model_id} ...")
from huggingface_hub import snapshot_download
ignore = ["*.onnx", "*.onnx_data", "onnx/*"] if "cross-encoder" in model_id else None
snapshot_download(repo_id=model_id, local_files_only=False, ignore_patterns=ignore)

# 4. Final post-download verification
if heal_and_verify(model_id):
    print(f"  [OK] Downloaded and verified: {model_id}")
else:
    print(f"  [ERROR] Snapshot verification failed after download: {model_id}")
    sys.exit(1)
EOF
}

ensure_venv() {
    if [ ! -d ".venv" ] || [ ! -x ".venv/bin/python3" ]; then
        info "Setting up Python 3.12 virtual environment (.venv) using uv ..."
        command -v uv &>/dev/null || error "'uv' is required to set up .venv, but was not found in PATH or ~/.local/bin."
        uv venv --python 3.12 .venv
        uv sync
        if [ -f "requirements.txt" ]; then
            uv pip install -r requirements.txt
        fi
        success "Virtual environment (.venv) initialized."
    fi
}

prepare_hf_models() {
    info "=== [1/3] Preparing & Verifying HuggingFace and PyTorch Models ==="
    ensure_venv
    mkdir -p .cache/huggingface/hub .cache/torch/hub/checkpoints

    prepare_pytorch_regnet
    prepare_hf_model "$HF_RERANKER"
    prepare_hf_model "$HF_NEMOTRON"
}

# =============================================================================
# PHASE 2 — Ollama Models
# =============================================================================

_wait_for_ollama() {
    local container="$1"
    info "Waiting for Ollama to become healthy ..."
    for _ in $(seq 1 30); do
        docker exec "$container" ollama list &>/dev/null && return
        sleep 2
    done
    error "Container '$container' is not responding. Check: docker compose up -d local-AI"
}

prepare_ollama_models() {
    info "=== [2/3] Preparing Ollama Models ==="
    command -v docker &>/dev/null || error "docker CLI not found."
    mkdir -p ollama-models/models

    if [ "$VERIFY_ONLY" = true ]; then
        [ -n "$(ls -A ollama-models/models/manifests 2>/dev/null)" ] \
            && success "Ollama models directory verified: ./ollama-models/models/ ($(du -sh ollama-models/models | cut -f1))" \
            || error "Verification failed: ollama-models/models/manifests is missing or empty."
        return
    fi

    local container="localAI"

    # Start container if not already running
    if ! docker inspect "$container" &>/dev/null; then
        info "Container '$container' not found — starting via docker compose ..."
        docker compose up -d local-AI
        _wait_for_ollama "$container"
    fi

    docker exec "$container" ollama list &>/dev/null \
        || error "Container '$container' is not responding."
    success "Container '$container' is active"

    local installed
    installed="$(docker exec "$container" ollama list 2>/dev/null || true)"

    for model in "${OLLAMA_MODELS[@]}"; do
        if [ "$FORCE_ALL" = false ] && echo "$installed" | grep -Eq "^${model}[[:space:]]"; then
            success "Ollama model already cached: $model"
        else
            info "Pulling model: $model ..."
            docker exec "$container" ollama pull "$model"
            success "Model ready: $model"
        fi
    done

    # Sync models to host bind-mount (docker-compose: ./ollama-models/models → /root/.ollama/models)
    if [ -n "$(ls -A ollama-models/models/manifests 2>/dev/null)" ]; then
        success "Ollama models synced via bind mount: ./ollama-models/models/ ($(du -sh ollama-models/models | cut -f1))"
    else
        info "Exporting Ollama models from container to ./ollama-models/models/ ..."
        docker cp "${container}:/root/.ollama/models/." ollama-models/models/ 2>/dev/null || true
        success "Ollama models exported to ./ollama-models/models/"
    fi
}

# =============================================================================
# PHASE 3 — Docker Images (Build & Export)
# =============================================================================

_save_image_if_needed() {
    local image="$1"
    local tar_path="$2"

    if [ -f "$tar_path" ] && [ "$FORCE_ALL" = false ]; then
        success "Image archive exists: $tar_path ($(du -sh "$tar_path" | cut -f1))"
        return
    fi

    [ "$VERIFY_ONLY" = true ] && error "Verification failed: tarball missing: $tar_path"

    info "Saving: $image → $tar_path"
    docker save "$image" -o "$tar_path"
    success "Saved: $tar_path ($(du -sh "$tar_path" | cut -f1))"
}

_generate_load_script() {
    cat > docker-images/load-images.sh << 'LOADSCRIPT'
#!/usr/bin/env bash
# Run this on the AIR-GAPPED machine to load all Docker images into local daemon.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Loading Docker images from $SCRIPT_DIR ..."
for TAR in "$SCRIPT_DIR"/*.tar; do
    [ -f "$TAR" ] || continue
    echo "  Loading: $(basename "$TAR") ..."
    docker load -i "$TAR"
done
echo "All images loaded. Verify with: docker images"
LOADSCRIPT
    chmod +x docker-images/load-images.sh
    success "Generated docker-images/load-images.sh"
}

prepare_docker_images() {
    info "=== [3/3] Building & Exporting Docker Images ==="
    command -v docker &>/dev/null || error "docker CLI not found."
    mkdir -p docker-images

    # 3a. Third-party base images
    local base_images=(
        "ollama/ollama:${OLLAMA_VERSION}"
        "neo4j:${NEO4J_VERSION}"
    )
    for image in "${base_images[@]}"; do
        local safe_name="${image//[\/:]/_}"
        local tar_path="docker-images/${safe_name}.tar"
        if [ -f "docker-images/${image}.tar" ] && [ ! -f "$tar_path" ]; then
            mv "docker-images/${image}.tar" "$tar_path"
        fi
        if [ "$VERIFY_ONLY" = true ]; then
            [ -f "$tar_path" ] || error "Verification failed: $tar_path missing"
            success "Base image archive verified: $tar_path"
        else
            [ -f "$tar_path" ] && [ "$FORCE_ALL" = false ] || docker pull "$image"
            _save_image_if_needed "$image" "$tar_path"
        fi
    done

    # 3b. Application images
    if [ "$VERIFY_ONLY" = true ]; then
        [ -f "docker-images/lolly-rag-backend.tar" ]  || error "Verification failed: backend tar missing"
        [ -f "docker-images/lolly-rag-frontend.tar" ] || error "Verification failed: frontend tar missing"
        success "Application image archives verified"
    else
        info "Building application images via Docker Compose ..."
        docker compose build backend-server frontend

        _save_image_if_needed "lolly-rag-backend:latest"  "docker-images/lolly-rag-backend.tar"
        _save_image_if_needed "lolly-rag-frontend:latest" "docker-images/lolly-rag-frontend.tar"
    fi

    _generate_load_script
}

# =============================================================================
# PHASE 4 — Bare-Metal Python Virtual Environment (.venv)
# =============================================================================

prepare_bare_metal() {
    info "=== [Optional] Preparing Python virtual environment for bare-metal (.venv) ==="
    command -v uv &>/dev/null || error "'uv' is required for bare-metal setup but not found in PATH or ~/.local/bin."

    if [ "$VERIFY_ONLY" = true ]; then
        [ -d ".venv" ] && [ -x ".venv/bin/python3" ] || error "Verification failed: .venv virtual environment missing or invalid"
        success "Virtual environment verified: .venv/ ($(du -sh .venv | cut -f1))"
        return
    fi

    if [ ! -d ".venv" ] || [ "$FORCE_ALL" = true ]; then
        info "Creating Python 3.12 virtual environment with uv ..."
        uv venv --python 3.12 .venv
    fi

    info "Syncing dependencies into .venv via uv sync ..."
    uv sync

    if [ -f "requirements.txt" ]; then
        info "Ensuring dependencies from requirements.txt are installed in .venv ..."
        uv pip install -r requirements.txt
    fi

    success "Bare-metal virtual environment (.venv) synced and ready ($(du -sh .venv | cut -f1))"
}

# Alias for backwards compatibility
prepare_wheels() {
    prepare_bare_metal
}

# =============================================================================
# SUMMARY
# =============================================================================

print_summary() {
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
    [ "$SKIP_BARE_METAL" = false ] && [ -d ".venv" ] && echo "    ✅ .venv/                   ($(du -sh .venv | cut -f1)) — Bare-metal Python virtualenv"
    echo ""
    echo "  Transfer Instructions:"
    echo "    1. Copy this entire project directory to the air-gapped machine."
    echo "    2. On the air-gapped machine, run either:"
    echo "         ./run-airgap-docker.sh   # For Docker deployment"
    echo "         ./start.sh               # For Bare-Metal deployment (uses .venv)"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

parse_args "$@"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Lolly RAG — Optimized Air-Gap Preparation            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
info "Active mode: $MODE  |  Force: $FORCE_ALL  |  Verify-only: $VERIFY_ONLY"
[ "$MODE" = "docker" ] && info "Docker mode: Python deps are packaged inside Docker images."
echo ""

[ "$SKIP_HF"         = false ] && prepare_hf_models     || warn "Skipping HuggingFace and PyTorch models (--skip-hf)"
[ "$SKIP_OLLAMA"     = false ] && prepare_ollama_models  || warn "Skipping Ollama models (--skip-ollama)"
[ "$SKIP_DOCKER"     = false ] && prepare_docker_images  || warn "Skipping Docker images (--skip-docker)"
[ "$SKIP_BARE_METAL" = false ] && prepare_bare_metal

print_summary
