#!/usr/bin/env bash
#
# ship_efficienttam_offline.sh — build an EfficientTAM offline bundle here
# (internet-connected machine) and ship it to an AIR-GAPPED server via rsync/scp.
# ========================================================================
# The server has no internet, so it cannot `pip install` or download the
# checkpoint. This script collects everything EfficientTAM needs:
#   - the 5 pure-python runtime wheels (hydra-core, omegaconf, antlr4-runtime,
#     iopath, portalocker)
#   - the EfficientTAM package source (editable-installable on the server)
#   - the efficienttam_s.pt checkpoint (~131 MB)
#   - install_on_server.sh — run it on the server, inside the target venv
# ...into one folder, then rsync/scp that folder to the server.
#
# Usage:
#   ./ship_efficienttam_offline.sh user@server-host [dest_dir]
#
#   dest_dir defaults to:
#     /home/ia/Documentos/hyperbolic_foveation_drones/efficienttam_telescope
#
# Idempotent: rebuilding reuses local files; rsync only sends what changed.

set -euo pipefail

BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}▸${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }
err()  { echo -e "${RED}✗${NC} $*"; }

SERVER="${1:-}"
DEST="${2:-/home/ia/Documentos/hyperbolic_foveation_drones/efficienttam_telescope}"

if [[ -z "$SERVER" ]]; then
    err "Missing server target."
    echo "usage: $0 user@server-host [dest_dir]"
    echo "  e.g.  $0 ia@10.0.0.42"
    echo "  dest_dir default: /home/ia/Documentos/hyperbolic_foveation_drones/efficienttam_telescope"
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${HERE}/efficienttam_offline_bundle"
WHEELS="${BUNDLE}/wheels"
PY="${HERE}/.telescope/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"

# Runtime deps + build backend (setuptools/wheel): a fresh py3.12 venv has no
# setuptools, and the editable install of EfficientTAM needs it as build backend.
DEPS=(hydra-core omegaconf antlr4-python3-runtime iopath portalocker setuptools wheel)

echo -e "\n${BOLD}== Building EfficientTAM offline bundle ==${NC}"
info "bundle dir : ${BUNDLE}"
info "target     : ${SERVER}:${DEST}"

mkdir -p "$WHEELS" "${BUNDLE}/checkpoints"

# ── 1. Runtime wheels ─────────────────────────────────────────────────────────
# Must be real wheels, not sdists: the air-gapped server cannot run a build step.
# antlr4-python3-runtime and iopath ship as sdists on PyPI, so we point pip at the
# curated wheels_offline/ (which has them pre-built) and force --prefer-binary.
info "Collecting runtime wheels (binary-only, no sdists) ..."
FL=()
[[ -d "${HERE}/wheels_offline" ]] && FL=(--find-links "${HERE}/wheels_offline")
if "$PY" -m pip download --quiet --dest "$WHEELS" --prefer-binary "${FL[@]}" "${DEPS[@]}"; then
    ok "Collected runtime wheels"
elif [[ -d "${HERE}/wheels_offline" ]]; then
    warn "pip download failed (offline here?) — copying matching wheels from wheels_offline/"
    cp "${HERE}/wheels_offline/"{hydra_core,omegaconf,antlr4_python3_runtime,iopath,portalocker,pyyaml,tqdm,packaging,typing_extensions}*.whl \
        "$WHEELS/" 2>/dev/null || true
    ok "Copied wheels from wheels_offline/"
else
    err "Could not obtain runtime wheels (no internet and no wheels_offline/)."
    exit 1
fi
# Safety net: strip any sdist that slipped in (the server can't build them offline).
if ls "$WHEELS"/*.tar.gz >/dev/null 2>&1; then
    warn "Dropping sdists from bundle (offline server can't build them): $(ls "$WHEELS"/*.tar.gz | xargs -n1 basename | tr '\n' ' ')"
    rm -f "$WHEELS"/*.tar.gz
fi

# ── 2. EfficientTAM package source ────────────────────────────────────────────
info "Staging EfficientTAM source ..."
if [[ -d "${HERE}/EfficientTAM" ]]; then
    rsync -a --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
        --exclude '*.pt' --exclude '*.pth' --exclude '.DS_Store' \
        "${HERE}/EfficientTAM/" "${BUNDLE}/EfficientTAM/"
    ok "Staged EfficientTAM/ source"
else
    info "EfficientTAM/ not present locally — cloning ..."
    git clone --depth 1 https://github.com/yformer/EfficientTAM "${BUNDLE}/EfficientTAM"
    rm -rf "${BUNDLE}/EfficientTAM/.git"
    ok "Cloned EfficientTAM/ source"
fi

# ── 3. Checkpoint (use local copy if we have it, else download) ────────────────
CKPT="${BUNDLE}/checkpoints/efficienttam_s.pt"
if [[ -f "$CKPT" ]]; then
    ok "Checkpoint already in bundle — skipping"
elif [[ -f "${HERE}/checkpoints/efficienttam_s.pt" ]]; then
    info "Copying local checkpoint into bundle ..."
    cp "${HERE}/checkpoints/efficienttam_s.pt" "$CKPT"
    ok "Checkpoint copied (131 MB)"
else
    info "Downloading efficienttam_s.pt (~131 MB, public) ..."
    "$PY" - "$CKPT" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="yunyangx/efficient-track-anything",
                    filename="efficienttam_s.pt")
shutil.copy(p, sys.argv[1])
print("OK", sys.argv[1])
PYEOF
    ok "Checkpoint downloaded"
fi

# ── 4. Server-side installer (runs offline, inside the server's venv) ──────────
cat > "${BUNDLE}/install_on_server.sh" <<'EOS'
#!/usr/bin/env bash
# Run ON THE SERVER, with your training venv ACTIVATED, from inside this folder:
#   source /path/to/your/.telescope/bin/activate
#   ./install_on_server.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing EfficientTAM offline from ${HERE}"
# Skip the CUDA `_C` extension: it's for mask/video post-processing, NOT the
# image encoder we use as a backbone — and compiling it with nvcc is slow
# (often looks like a hang). The backbone imports and runs fine without it.
export Efficient_Track_Anything_BUILD_CUDA=0
# Build backend first (fresh py3.12 venvs ship without setuptools).
pip install --no-index --find-links "${HERE}/wheels" setuptools wheel
pip install --no-index --find-links "${HERE}/wheels" \
    hydra-core omegaconf antlr4-python3-runtime iopath portalocker
pip install --no-index --no-build-isolation --no-deps -e "${HERE}/EfficientTAM"

python -c "import efficient_track_anything; print('efficient_track_anything import OK')"
echo
echo "Done. Checkpoint is at:"
echo "  ${HERE}/checkpoints/efficienttam_s.pt"
echo "Train with:  --backbone efficienttam --backbone_ckpt ${HERE}/checkpoints/efficienttam_s.pt"
EOS
chmod +x "${BUNDLE}/install_on_server.sh"
ok "Wrote install_on_server.sh"

BUNDLE_SIZE=$(du -sh "$BUNDLE" | cut -f1)
ok "Bundle ready (${BUNDLE_SIZE})"

# ── 5. Ship it ────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}== Transferring to ${SERVER}:${DEST} ==${NC}"
info "Creating remote dir ..."
ssh "$SERVER" "mkdir -p '${DEST}'"

if command -v rsync >/dev/null 2>&1; then
    info "rsync (resumable) ..."
    rsync -avP "${BUNDLE}/" "${SERVER}:${DEST}/"
else
    warn "rsync not found — falling back to scp"
    scp -r "${BUNDLE}/." "${SERVER}:${DEST}/"
fi

ok "Transfer complete."
echo
echo -e "${BOLD}Next, on the server:${NC}"
echo "  cd ${DEST}"
echo "  source /path/to/your/.telescope/bin/activate   # your training venv"
echo "  ./install_on_server.sh"
