#!/usr/bin/env bash
#
# install.sh — Telescope environment, models, and datasets installer
# ===================================================================
# Sets up everything needed to run Telescope:
#   1. Python virtual environment (.telescope)
#   2. Core + optional training dependencies
#   3. SAM 3.1 backbone (default) — or SAM 2.1 fallback
#   3b. EfficientTAM lightweight backbone (optional — edge / drones)
#   4. Argoverse 2 dataset (optional)
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# The script is idempotent: re-running it skips steps already done.

set -u   # treat unset variables as errors (but not -e: we handle errors ourselves)

# ── Pretty printing ───────────────────────────────────────────────────
BOLD='\033[1m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
err()   { echo -e "${RED}✗${NC} $*"; }
hdr()   { echo -e "\n${BOLD}== $* ==${NC}"; }

ask_yn() {  # ask_yn "Question?" → returns 0 for yes, 1 for no (default no)
    local prompt="$1" reply
    read -r -p "$(echo -e "${YELLOW}?${NC} ${prompt} [y/N] ")" reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.telescope"
CKPT_DIR="${PROJECT_DIR}/checkpoints"

cd "$PROJECT_DIR"

# ──────────────────────────────────────────────────────────────────────
hdr "1/5  Python virtual environment"
# ──────────────────────────────────────────────────────────────────────

if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found. Install Python 3.9+ first."
    exit 1
fi

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python ${PY_VER} detected"

if [[ -d "$VENV_DIR" ]]; then
    ok "venv already exists at .telescope"
else
    info "Creating virtual environment .telescope ..."
    python3 -m venv "$VENV_DIR" && ok "venv created" || { err "venv creation failed"; exit 1; }
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
ok "venv activated"

# ──────────────────────────────────────────────────────────────────────
hdr "2/5  Python dependencies"
# ──────────────────────────────────────────────────────────────────────

info "Upgrading pip ..."
pip install --quiet --upgrade pip

info "Installing core dependencies (requirements.txt) ..."
if pip install --quiet -r requirements.txt; then
    ok "Core dependencies installed"
else
    err "Core dependency install failed. See output above."
    exit 1
fi

info "Installing telescope package (editable) ..."
pip install --quiet -e . && ok "telescope installed" || { err "package install failed"; exit 1; }

if ask_yn "Install training extras now (transformers, av2, pycocotools)? Needed for train.py / notebook 06."; then
    info "Installing training extras (requirements-train.txt) ..."
    if pip install -r requirements-train.txt; then
        ok "Training extras installed"
    else
        warn "Some training extras failed to build (av2/pycocotools can be finicky)."
        warn "Notebooks 01–05 still work. Retry later with: pip install -r requirements-train.txt"
    fi
else
    info "Skipped training extras. Notebooks 01–05 will work without them."
fi

# ──────────────────────────────────────────────────────────────────────
hdr "3/5  SAM backbone checkpoint"
# ──────────────────────────────────────────────────────────────────────

mkdir -p "$CKPT_DIR"

SAM3_CKPT="${CKPT_DIR}/sam3.1_multiplex.pt"
SAM2_CKPT="${CKPT_DIR}/sam2.1_hiera_large.pt"
ET_CKPT="${CKPT_DIR}/efficienttam_s.pt"
ET_DIR="${PROJECT_DIR}/EfficientTAM"

download_sam3() {
    echo
    info "SAM 3.1 is a gated model. You must have:"
    info "  1. Filled the access form at https://huggingface.co/facebook/sam3.1"
    info "  2. Been approved by Meta (you receive an email)"
    echo
    if ! ask_yn "Have you been APPROVED for facebook/sam3.1 on HuggingFace?"; then
        return 1   # trigger fallback
    fi

    echo
    info "Paste a HuggingFace access token with 'Read access to public gated repos'."
    info "Create one at: https://huggingface.co/settings/tokens"
    read -r -s -p "$(echo -e "${YELLOW}?${NC} HF token (hidden): ")" HF_TOKEN
    echo
    if [[ -z "$HF_TOKEN" ]]; then
        warn "No token entered."
        return 1
    fi

    info "Downloading sam3.1_multiplex.pt (~3.5 GB) ..."
    if HF_TOKEN="$HF_TOKEN" python - <<'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(
        repo_id="facebook/sam3.1",
        filename="sam3.1_multiplex.pt",
        local_dir="checkpoints",
        token=os.environ["HF_TOKEN"],
    )
    print(f"OK {p}")
except Exception as e:
    print(f"ERROR {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        ok "SAM 3.1 checkpoint downloaded"
        # Also offer to clone the SAM3 code repo (needed to load the weights)
        if [[ ! -d "${PROJECT_DIR}/sam3" ]] && ask_yn "Clone + install the SAM3 code repo (needed to load weights)?"; then
            git clone https://github.com/facebookresearch/sam3 "${PROJECT_DIR}/sam3" \
                && pip install -e "${PROJECT_DIR}/sam3" \
                && ok "SAM3 code installed" \
                || warn "SAM3 code install failed — see github.com/facebookresearch/sam3"
        fi
        return 0
    else
        err "SAM 3.1 download failed (token wrong, or not yet approved?)."
        return 1
    fi
}

download_sam2() {
    echo
    info "Falling back to SAM 2.1 (public, no approval needed, same ViT-H backbone)."
    info "Downloading sam2.1_hiera_large.pt (~2.4 GB) from Meta CDN ..."
    if wget -q --show-progress -c \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        -O "$SAM2_CKPT"; then
        ok "SAM 2.1 checkpoint downloaded"
        pip install --quiet sam2 && ok "sam2 package installed" \
            || warn "sam2 package install failed — retry with: pip install sam2"
        return 0
    else
        err "SAM 2.1 download failed. Check your internet connection."
        return 1
    fi
}

# EfficientTAM: lightweight (~22M) edge/drone backbone. The package source lives
# in EfficientTAM/ (gitignored, so NOT pulled with the repo) and must be installed
# editable into THIS venv on every machine — see backbone_efficienttam.py / README.
setup_efficienttam() {
    # 1. Source package present? vendor it (gitignored → not pulled with the repo).
    if [[ -d "$ET_DIR" ]]; then
        ok "EfficientTAM source already present (EfficientTAM/)"
    else
        info "Fetching EfficientTAM source ..."
        git clone --depth 1 https://github.com/yformer/EfficientTAM "$ET_DIR" \
            || { err "EfficientTAM clone failed (need internet, or vendor EfficientTAM/ manually)."; return 1; }
    fi

    # 2. Runtime deps + editable install into the venv (so 'efficient_track_anything' imports).
    info "Installing EfficientTAM runtime deps + package (editable) ..."
    pip install --quiet hydra-core omegaconf iopath \
        && pip install --quiet -e "$ET_DIR" --no-build-isolation --no-deps \
        || { err "EfficientTAM package install failed."; return 1; }

    # 3. Checkpoint (public — no HF token needed).
    if [[ -f "$ET_CKPT" ]]; then
        ok "EfficientTAM checkpoint already present — skipping"
    else
        info "Downloading efficienttam_s.pt (~140 MB, public) ..."
        if python - <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(repo_id="yunyangx/efficient-track-anything",
                        filename="efficienttam_s.pt", local_dir="checkpoints")
    print(f"OK {p}")
except Exception as e:
    print(f"ERROR {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
        then
            ok "EfficientTAM checkpoint downloaded"
        else
            err "EfficientTAM checkpoint download failed."
            return 1
        fi
    fi

    # 4. Confirm the package actually imports (the step people forget on new machines).
    if python -c "import efficient_track_anything" 2>/dev/null; then
        ok "EfficientTAM ready (import OK)"
    else
        err "efficient_track_anything not importable after install."
        return 1
    fi
}

if [[ -f "$SAM3_CKPT" ]]; then
    ok "SAM 3.1 checkpoint already present — skipping"
elif [[ -f "$SAM2_CKPT" ]]; then
    ok "SAM 2.1 checkpoint already present — skipping"
else
    if ask_yn "Download a backbone checkpoint now?"; then
        # Try SAM 3.1 first (default), fall back to SAM 2.1
        if ! download_sam3; then
            warn "Using SAM 2.1 fallback."
            download_sam2 || warn "No backbone downloaded. The stub model still runs."
        fi
    else
        info "Skipped backbone download. The stub model runs without it."
    fi
fi

# ── EfficientTAM lightweight backbone (edge / drones) ──────────────────
echo
if [[ -f "$ET_CKPT" ]] && python -c "import efficient_track_anything" 2>/dev/null; then
    ok "EfficientTAM backbone already set up — skipping"
elif ask_yn "Set up the EfficientTAM lightweight backbone too (edge / drone training)?"; then
    setup_efficienttam || warn "EfficientTAM setup incomplete — see messages above."
else
    info "Skipped EfficientTAM. (SAM backbone or the stub still works.)"
fi

# ──────────────────────────────────────────────────────────────────────
hdr "4/5  Argoverse 2 dataset (optional)"
# ──────────────────────────────────────────────────────────────────────

DATA_DIR="${PROJECT_DIR}/data/argoverse2"

# Install s5cmd (fast parallel S3 downloader) into the venv if not present.
ensure_s5cmd() {
    command -v s5cmd >/dev/null 2>&1 && return 0
    [[ -x "${VENV_DIR}/bin/s5cmd" ]] && return 0
    info "Installing s5cmd (fast S3 downloader) ..."
    local tmp
    tmp=$(mktemp -d)
    if wget -q --show-progress \
           "https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz" \
           -O "${tmp}/s5cmd.tar.gz" \
        && tar xzf "${tmp}/s5cmd.tar.gz" -C "$tmp" s5cmd \
        && mv "${tmp}/s5cmd" "${VENV_DIR}/bin/s5cmd"; then
        ok "s5cmd installed"
        rm -rf "$tmp"
    else
        err "Failed to install s5cmd. Check your internet connection."
        rm -rf "$tmp"
        return 1
    fi
}

# Print a numbered menu and return "<n_train> <n_val>".
# All display output goes to stderr so it isn't captured when called inside $().
# read uses /dev/tty directly so it works even in a subshell.
ask_subset() {
    echo -e "${YELLOW}?${NC} How much data to download?" >&2
    echo "  Each 'log' is one ~20 s drive segment with LiDAR sweeps, 9 cameras, and 3-D box annotations." >&2
    echo >&2
    echo "  1) Small sample   —   5 train +   5 val  (~5 GB)   Verify the pipeline runs; not enough to train." >&2
    echo "  2) Medium subset  —  20 train +  10 val  (~20 GB)  Quick training run; expect lower mAP than full." >&2
    echo "  3) Large subset   —  50 train +  20 val  (~60 GB)  Decent experiments; ~7 % of full train split." >&2
    echo "  4) Full dataset   — 700 train + 150 val (~500 GB)  Complete split for publication-quality results." >&2
    printf '%b' "${YELLOW}  Choice [1-4, default 2]: ${NC}" >&2
    local choice
    IFS= read -r choice < /dev/tty
    case "$choice" in
        1) echo "5 5" ;;
        3) echo "50 20" ;;
        4) echo "700 150" ;;
        *) echo "20 10" ;;
    esac
}

# Download n_train train logs and n_val val logs from the public S3 bucket.
download_av2_subset() {
    local n_train="$1" n_val="$2"
    local s5cmd_bin
    s5cmd_bin=$(command -v s5cmd 2>/dev/null || echo "${VENV_DIR}/bin/s5cmd")

    local train_ids val_ids
    train_ids=$(python - <<PYEOF
from av2.datasets.sensor.splits import TRAIN
print('\n'.join(TRAIN[:${n_train}]))
PYEOF
)
    val_ids=$(python - <<PYEOF
from av2.datasets.sensor.splits import VAL
print('\n'.join(VAL[:${n_val}]))
PYEOF
)

    local total=$(( n_train + n_val ))
    info "Downloading ${n_train} train + ${n_val} val logs (${total} total) ..."

    local failed=0

    while IFS= read -r log_id; do
        [[ -z "$log_id" ]] && continue
        local dest="${DATA_DIR}/sensor/train/${log_id}"
        if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
            info "  [skip] train/${log_id}"
            continue
        fi
        mkdir -p "$dest"
        info "  train/${log_id}"
        "$s5cmd_bin" --no-sign-request cp \
            "s3://argoverse/datasets/av2/sensor/train/${log_id}/*" "${dest}/" \
            || { warn "  Failed: train/${log_id}"; failed=$((failed + 1)); }
    done <<< "$train_ids"

    while IFS= read -r log_id; do
        [[ -z "$log_id" ]] && continue
        local dest="${DATA_DIR}/sensor/val/${log_id}"
        if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
            info "  [skip] val/${log_id}"
            continue
        fi
        mkdir -p "$dest"
        info "  val/${log_id}"
        "$s5cmd_bin" --no-sign-request cp \
            "s3://argoverse/datasets/av2/sensor/val/${log_id}/*" "${dest}/" \
            || { warn "  Failed: val/${log_id}"; failed=$((failed + 1)); }
    done <<< "$val_ids"

    if (( failed > 0 )); then
        warn "${failed} log(s) failed. Re-run install.sh to retry (skips already-downloaded logs)."
        return 1
    fi
    ok "Argoverse 2 subset downloaded (${n_train} train + ${n_val} val)"
}

n_train_present=$(find "${DATA_DIR}/sensor/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if (( n_train_present > 0 )); then
    ok "Argoverse 2 data already present (${n_train_present} train logs at data/argoverse2/sensor/train)"
elif ask_yn "Download Argoverse 2 dataset? (only needed for training)"; then
    if ! python -c "import av2" 2>/dev/null; then
        warn "av2 not installed. Install training extras first, then re-run."
    elif ensure_s5cmd; then
        read -r n_train n_val <<< "$(ask_subset)"
        mkdir -p "${DATA_DIR}/sensor"
        download_av2_subset "$n_train" "$n_val"
    fi
else
    info "Skipped dataset download. You can train later once data is available."
fi

# ──────────────────────────────────────────────────────────────────────
hdr "5/5  Verification"
# ──────────────────────────────────────────────────────────────────────

info "Running a quick package self-test ..."
if python - <<'PYEOF'
import torch
from telescope import TelescopeModel, HungarianMatcher, match_and_compute_loss
m = TelescopeModel(num_classes=7, num_queries=10, query_dim=32, enc_out_dim=32)
with torch.no_grad():
    boxes, logits, o, R = m(torch.rand(1, 3, 64, 64))
assert boxes.shape == (1, 10, 4)
print("self-test OK")
PYEOF
then
    ok "Package self-test passed"
else
    err "Package self-test failed — see output above"
    exit 1
fi

# ── Summary ────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}${GREEN}Installation complete.${NC}"
echo
echo "Next steps:"
echo "  source .telescope/bin/activate     # activate the environment"
echo "  jupyter notebook                   # open notebooks 01–05 (learning path)"
echo
[[ -f "$SAM3_CKPT" ]] && echo "  Backbone: SAM 3.1       (checkpoints/sam3.1_multiplex.pt)"
[[ -f "$SAM2_CKPT" ]] && echo "  Backbone: SAM 2.1       (checkpoints/sam2.1_hiera_large.pt)"
[[ -f "$ET_CKPT"   ]] && echo "  Backbone: EfficientTAM  (checkpoints/efficienttam_s.pt)"
echo
echo "  To train on Argoverse 2 once data is ready:"
echo "    python train.py --data_dir ./data/argoverse2/sensor/train \\"
echo "                    --val_dir  ./data/argoverse2/sensor/val --fp16"
if [[ -f "$ET_CKPT" ]]; then
    echo
    echo "  To train on a YOLO-format drone dataset with the lightweight backbone:"
    echo "    python train.py --dataset drones --data_dir <yolo_root> --val_dir <yolo_root> \\"
    echo "                    --backbone efficienttam --backbone_ckpt ./checkpoints/efficienttam_s.pt \\"
    echo "                    --image_size 1024 1024 --batch_size 4 --epochs 4 --fp16"
fi
echo
