# Telescope — Learnable Hyperbolic Foveation

Implementation of **"Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection"**  
Ewen et al., 2026 · [arXiv:2604.06332](https://arxiv.org/abs/2604.06332) · [Project page](https://light.princeton.edu/telescope)

---

## Installation

### Quick start — one script does everything

```bash
git clone https://github.com/your-user/telescope
cd telescope
chmod +x install.sh
./install.sh
```

`install.sh` is interactive and handles the whole setup:
1. Creates the `.telescope` virtual environment
2. Installs core dependencies (and optionally the training extras)
3. Downloads a backbone — **asks if you've been approved for SAM 3.1**; if yes it asks for
   your HuggingFace token and downloads it, otherwise it falls back to SAM 2.1 automatically
4. Optionally downloads the Argoverse 2 dataset
5. Runs a self-test to confirm everything works

It is safe to re-run — completed steps are skipped.

### Manual install (if you prefer)

```bash
python -m venv .telescope
source .telescope/bin/activate        # Windows: .telescope\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt       # core: enough for notebooks 01–06
pip install -e .
pip install -r requirements-train.txt # extras: only for train.py / eval.py on real data
```

| File | When you need it |
|---|---|
| `requirements.txt` | always — runs the package and all notebooks |
| `requirements-train.txt` | only to train/evaluate on real data (`transformers`, `av2`, `pycocotools`) |

### Backbone & data details

The `install.sh` script automates the steps below; they are documented here for reference.

<details>
<summary><b>SAM 3.1 backbone (gated — needs Meta approval)</b></summary>

1. Request access at **https://huggingface.co/facebook/sam3.1** (fill the form → Meta approves).
2. Create a **classic** Read token at huggingface.co/settings/tokens (or a fine-grained token
   with *"Access public gated repos"* enabled), then `hf auth login`.
3. Download (~3.5 GB) — note the filename is `sam3.1_multiplex.pt`:

```bash
git clone https://github.com/facebookresearch/sam3 && pip install -e sam3
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="facebook/sam3.1", filename="sam3.1_multiplex.pt", local_dir="checkpoints")
EOF
```
</details>

<details>
<summary><b>SAM 2.1 fallback (public, no approval)</b></summary>

```bash
pip install sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O checkpoints/sam2.1_hiera_large.pt
```
Same ViT-H backbone family — a drop-in substitute while waiting for SAM 3.1 approval.
</details>

<details>
<summary><b>Argoverse 2 dataset (TruckDrive substitute)</b></summary>

TruckDrive (the paper's dataset, up to 1 km) is not yet public. Argoverse 2 covers up to
~250 m in the same format:

```bash
pip install av2
python -m av2.datasets.sensor.download --target_dir ./data/argoverse2
```
</details>

---

## Notebooks (the learning path)

Six notebooks tell one continuous story — *building a digital telescope*. Run them in order;
each one explains how its step builds on the last. They are written to be readable by
non-experts while keeping the real maths and code.

```bash
source .telescope/bin/activate
jupyter notebook
```

| # | Notebook | The story step | What you learn |
|---|---|---|---|
| 01 | `01_geometric_engine.ipynb` | Grind the lens | Φ, Φ⁻¹ (Newton-Raphson), Jacobian, Riemannian boxes |
| 02 | `02_foveation_warp.ipynb` | Mount the lens | Differentiable image warp with `grid_sample` |
| 03 | `03_hyperbolic_embedding.ipynb` | Calibrate it | Telling the detector the warp settings |
| 04 | `04_detection_head.ipynb` | Read through it | Box head, gIoU, loss, denoising |
| 05 | `05_full_pipeline.ipynb` | Assemble it | Full model, Hungarian matching, inference |
| 06 | `06_results_analysis.ipynb` | Test if it sees farther | Metrics, distance plots, with-vs-without comparison |

> **Notebooks 01–05** run on any laptop CPU (they use tiny stand-in models).
> **Notebook 06** is the post-training analysis — it shows real results once you've trained,
> and runs in a clearly-labelled demo mode before then so you can preview the analysis.

---

## Training

### Single GPU (RTX 3500 Ada, 14 GB)

```bash
python train.py \
    --data_dir ./data/argoverse2/sensor/train \
    --val_dir  ./data/argoverse2/sensor/val \
    --batch_size 2 \
    --fp16
```

### 2× GPU — DDP (server with 2× 24 GB)

```bash
torchrun --nproc_per_node=2 train.py \
    --data_dir ./data/argoverse2/sensor/train \
    --val_dir  ./data/argoverse2/sensor/val \
    --batch_size 4 \
    --fp16
```

### Resume from checkpoint

```bash
python train.py \
    --data_dir ./data/argoverse2/sensor/train \
    --val_dir  ./data/argoverse2/sensor/val \
    --resume   ./runs/run_01 \
    --fp16
```

### With SAM3.1 backbone (once downloaded)

```bash
python train.py \
    --data_dir       ./data/argoverse2/sensor/train \
    --val_dir        ./data/argoverse2/sensor/val \
    --backbone_ckpt  ./checkpoints/sam3.1_multiplex.pt \
    --fp16
```

### Key hyperparameters (paper defaults)

| Parameter | Default | Notes |
|---|---|---|
| `--lr` | 1e-4 | AdamW, lambda schedule with 1 warm-up epoch |
| `--batch_size` | 4 | Per GPU. Use 2 on 14 GB VRAM |
| `--epochs` | 12 | Fine-tune from backbone checkpoint |
| `--image_size` | 1024 1024 | Paper resolution |
| `--num_queries` | 300 | DETR object queries |
| `--query_dim` | 256 | Query / embedding dimension |
| `--fp16` | flag | Mixed precision — strongly recommended |

---

## Validation

Run on the val split and print COCO mAP:

```bash
python eval.py \
    --data_dir  ./data/argoverse2/sensor/val \
    --checkpoint ./runs/run_01/checkpoint_best.pt
```

Expected metrics on Argoverse 2 (from the paper, TruckDrive numbers):

| Method | mAP | mAP₅₀ | mAP₀₋₅₀ | mAP₅₀₋₁₅₀ | mAP₁₅₀₋₂₅₀ | mAP₂₅₀₊ |
|---|---|---|---|---|---|---|
| Deformable DETR | 0.166 | 0.335 | 0.396 | 0.178 | 0.081 | 0.072 |
| DINO | 0.222 | 0.405 | 0.335 | 0.239 | 0.189 | 0.179 |
| **Telescope (ours)** | **0.497** | **0.801** | **0.608** | **0.507** | **0.335** | **0.326** |

---

## Test

Run on the test split:

```bash
python eval.py \
    --data_dir   ./data/argoverse2/sensor/test \
    --checkpoint ./runs/run_01/checkpoint_best.pt \
    --split test
```

Results are saved to `runs/run_01/test_results.json` in COCO format.

---

## Comparison: Telescope vs baseline (no foveation)

To train and compare a baseline Deformable DETR **without** the hyperbolic foveation layer:

```bash
# Baseline: disable foveation (R → 0 makes Phi = identity everywhere)
python train.py \
    --data_dir    ./data/argoverse2/sensor/train \
    --val_dir     ./data/argoverse2/sensor/val \
    --output_dir  ./runs/baseline \
    --no_foveation \
    --fp16

# Telescope (default)
python train.py \
    --data_dir   ./data/argoverse2/sensor/train \
    --val_dir    ./data/argoverse2/sensor/val \
    --output_dir ./runs/telescope \
    --fp16

# Compare
python compare.py \
    --runs ./runs/baseline ./runs/telescope \
    --labels "No foveation" "Telescope"
```

The `--no_foveation` flag fixes R to a near-zero constant so `w(r) = 0`
everywhere and `Φ(x) = x` — the model becomes standard Deformable DETR
with the same backbone, making the comparison ablation fair.

> **Prefer an interactive analysis?** Open **`notebooks/06_results_analysis.ipynb`**. It
> loads your trained checkpoint(s), computes metrics, plots accuracy **by distance**, and
> shows the with-vs-without-foveation comparison with rendered detections. It even runs
> before you've trained (in a clearly-labelled demo mode) so you can preview the analysis.

---

## Package structure

```
telescope/                  the importable package (single source of truth)
├── geometry.py     Φ, Φ⁻¹, J_Φ, validate_inversion
├── box.py          Euclidean ↔ Riemannian box encode/decode
├── warp.py         FoveationWarpLayer
├── estimator.py    FoveationEstimator FFN
├── embedding.py    HyperbolicEmbedding + augment_queries
├── head.py         RiemannianBoxHead, TelescopeLoss, gIoU, denoise_boxes
├── matcher.py      HungarianMatcher, match_and_compute_loss
├── eval.py         CocoEvaluator (mAP, mAP@50, per-distance bins)
├── data.py         Argoverse2Dataset, collate_fn
├── checkpoint.py   CheckpointManager (save/load/best-tracking/rotation)
└── pipeline.py     TelescopeModel (full two-stage system)

notebooks/                  the learning path (01–06, import from telescope/)

train.py                    training script (single GPU + DDP, FP16, --no_foveation)
eval.py                     evaluation script (COCO mAP)
compare.py                  baseline comparison plots
install.sh                  interactive installer (env + models + data)
requirements.txt            core dependencies
requirements-train.txt      training/eval extras
```

---

## Hardware

| Mode | GPU | Notes |
|---|---|---|
| Notebooks (stub) | CPU | No GPU needed |
| Inference FP16 | 4 GB VRAM | Any modern GPU |
| Training FP16, batch 2 | ~12 GB VRAM | RTX 3500 Ada 14 GB ✓ |
| Training FP16, batch 4 | ~16 GB VRAM | RTX 3500 Ada (tight — use `--grad_accum 2`) |
| Training FP16, batch 4 × 2 GPU | ~16 GB / GPU | 2 × 24 GB ✓ |

---

## Citation

```bibtex
@article{ewen2026telescope,
  title   = {Telescope: Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection},
  author  = {Ewen, Parker and Rivkin, Dmitriy and Bijelic, Mario and Heide, Felix},
  journal = {arXiv:2604.06332},
  year    = {2026}
}
```
