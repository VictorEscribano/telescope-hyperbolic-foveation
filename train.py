"""
train.py — Telescope training script
=====================================

Usage (single GPU):
    python train.py --data_dir ./data/argoverse2/sensor/train \
                    --val_dir  ./data/argoverse2/sensor/val   \
                    --output_dir ./runs/run_01

Usage (2 GPU DDP — for the 2×24GB server):
    torchrun --nproc_per_node=2 train.py \
        --data_dir ./data/argoverse2/sensor/train \
        --val_dir  ./data/argoverse2/sensor/val   \
        --output_dir ./runs/run_01

Key hyperparameters match paper Table 9:
    lr=1e-4, batch=4, epochs=12, image_size=1024
"""

import argparse
import os
import time
from pathlib import Path

# Reduce allocator fragmentation before any CUDA tensors are created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

# Input shapes are static here (fixed --image_size and --batch_size), so let
# cuDNN benchmark/autotune the best conv algorithms once and reuse them. And on
# Ampere+ (e.g. A10) allow TF32 for the fp32 matmuls in the geometry path — the
# detector runs in fp16 and the SAM3 backbone in bf16, so neither is affected;
# only the otherwise-slow fp32 Newton-Raphson/Jacobian matmuls speed up.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from telescope.pipeline import TelescopeModel
from telescope.matcher import (HungarianMatcher, match_and_compute_loss,
                               compute_denoising_loss, compute_encoder_aux_loss)
from telescope.eval import CocoEvaluator, DetectionResult, DISTANCE_BINS
from telescope.data import collate_fn
from telescope.checkpoint import CheckpointManager


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     type=str, default="argoverse2",
                   choices=["argoverse2", "drones"],
                   help="argoverse2 (3D→2D) or drones (YOLO-format 2D). For "
                        "drones, point --data_dir and --val_dir both at the "
                        "dataset root (the loader picks the train/ and val/ subdirs).")
    p.add_argument("--data_dir",    type=str, required=True)
    p.add_argument("--val_dir",     type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="./runs/run_01")
    p.add_argument("--epochs",      type=int, default=12)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--image_size",  type=int, nargs=2, default=[1024, 1024])
    p.add_argument("--num_queries", type=int, default=300)
    p.add_argument("--query_dim",   type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16",        action="store_true", default=True)
    p.add_argument("--grad_clip",   type=float, default=0.1)
    p.add_argument("--resume",      type=str, default=None,
                   help="path to checkpoint dir to resume from")
    p.add_argument("--backbone", type=str, default="sam3",
                   choices=["sam3", "efficienttam"],
                   help="which frozen backbone to load when --backbone_ckpt is given. "
                        "sam3 = SAM 3.1 (453M, max accuracy); efficienttam = "
                        "EfficientTAM ViT (~10-40× lighter, edge real-time).")
    p.add_argument("--backbone_ckpt", type=str, default=None,
                   help="path to the backbone checkpoint (optional — uses stub if not "
                        "given). SAM3.1: sam3.1_multiplex.pt; EfficientTAM: efficienttam_s.pt")
    p.add_argument("--et_config", type=str,
                   default="configs/efficienttam/efficienttam_s.yaml",
                   help="EfficientTAM Hydra config (variant): *_s.yaml (1024, accuracy) "
                        "or *_s_512x512.yaml / *_ti*.yaml (faster, edge)")
    p.add_argument("--no_foveation", action="store_true", default=False,
                   help="disable foveation (R fixed near zero) for ablation baseline")
    p.add_argument("--grad_accum",   type=int, default=1,
                   help="gradient accumulation steps (use 2 on 14GB VRAM for effective batch=4)")
    p.add_argument("--denoising",    action="store_true", default=True,
                   help="DINO-style denoising auxiliary loss (paper §4)")
    p.add_argument("--no_denoising", dest="denoising", action="store_false",
                   help="disable the denoising auxiliary loss")
    p.add_argument("--dn_noise_scale", type=float, default=0.4,
                   help="box-relative Gaussian noise std for denoising queries")
    p.add_argument("--dn_weight",    type=float, default=1.0,
                   help="weight of the denoising loss in the total")
    p.add_argument("--two_stage",    dest="two_stage", action="store_true", default=True,
                   help="DINO two-stage query selection from encoder proposals (paper Table 9)")
    p.add_argument("--no_two_stage", dest="two_stage", action="store_false",
                   help="disable two-stage; use 300 learned object queries (one-stage)")
    p.add_argument("--enc_weight",   type=float, default=1.0,
                   help="weight of the two-stage encoder auxiliary loss")
    return p.parse_args()


# ── DDP setup ─────────────────────────────────────────────────────────────────

def setup_ddp():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    rank, world_size, device = setup_ddp()
    is_main    = (rank == 0)

    # ── Dataset + class set selection ─────────────────────────────────────────
    if args.dataset == "drones":
        from telescope.data_drones import (DronesYoloDataset as DatasetCls,
                                            DRONE_NUM_CLASSES as NUM_CLASSES,
                                            DRONE_CLASS_NAMES as CLASS_NAMES)
    else:
        from telescope.data import (Argoverse2Dataset as DatasetCls,
                                     NUM_CLASSES, CLASS_NAMES)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TelescopeModel(
        num_classes = NUM_CLASSES,
        num_queries = args.num_queries,
        query_dim   = args.query_dim,
        two_stage   = args.two_stage,
    ).to(device)

    # Baseline ablation: fix R ≈ 0 so Phi(x) = x everywhere
    if args.no_foveation:
        print("[baseline] foveation disabled — R fixed to 0.001 (identity warp)")
        for param in model.fov_estimator.parameters():
            param.requires_grad_(False)
        model.fov_estimator._no_foveation = True   # checked in forward below

    # Optionally load a real (frozen) backbone in place of the stub.
    if args.backbone_ckpt:
        if args.backbone == "efficienttam":
            _load_efficienttam_backbone(model, args.backbone_ckpt, device, args.et_config)
        else:
            _load_sam3_backbone(model, args.backbone_ckpt, device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    if world_size > 1:
        # find_unused_parameters: two-stage leaves the learned object_queries /
        # ref_pts (requires_grad=True but unused — queries come from encoder
        # proposals) without a gradient, which DDP rejects unless told to expect it.
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                   weight_decay=args.weight_decay)
    # init_scale below the 2**16 default: the geometry runs in fp32 but the DETR
    # is fp16, and a lower starting scale avoids a few wasted overflow steps
    # while the scaler settles (it still adapts up/down as needed).
    scaler    = GradScaler("cuda", enabled=args.fp16, init_scale=2**13)

    # ── Checkpoint manager ────────────────────────────────────────────────────
    ckpt_mgr    = CheckpointManager(args.output_dir, keep_last=3)
    start_epoch = 0
    if args.resume:
        ckpt_mgr_resume = CheckpointManager(args.resume)
        start_epoch     = ckpt_mgr_resume.load_latest(
            model if world_size == 1 else model.module,
            optimizer, scaler, device=str(device)
        )

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_ds = DatasetCls(args.data_dir, split="train",
                          image_size=tuple(args.image_size))
    val_ds   = DatasetCls(args.val_dir,  split="val",
                          image_size=tuple(args.image_size))

    train_sampler = DistributedSampler(train_ds) if world_size > 1 else None
    train_loader  = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── LR warm-up (paper Table 9: 1 warm-up epoch) ───────────────────────────
    # Implemented as a per-iteration linear ramp over the first epoch, then
    # constant.  The previous per-epoch LambdaLR evaluated the ramp at epoch
    # index 0 → factor 1e-8, i.e. the entire first epoch trained at ~0 LR.
    warmup_iters = len(train_loader)
    def set_lr(global_step: int) -> float:
        scale = min(1.0, (global_step + 1) / max(1, warmup_iters))
        lr = args.lr * scale
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr
    global_step = start_epoch * len(train_loader)

    # ── Matching + evaluation ─────────────────────────────────────────────────
    matcher = HungarianMatcher(cost_cls=1.0, cost_l1=5.0, cost_giou=2.0)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            gt_boxes_list  = [t["boxes"].to(device)  for t in targets]
            gt_labels_list = [t["labels"].to(device) for t in targets]

            # gradient accumulation: zero_grad only at start of accumulation window
            if step % args.grad_accum == 0:
                optimizer.zero_grad()

            with autocast("cuda", enabled=args.fp16):
                _model = model.module if world_size > 1 else model

                # baseline ablation (--no_foveation) is handled inside the model
                # forward, which forces R≈0 so warp, embedding, and decode all
                # use the identity transform consistently.
                dn_in = ((gt_boxes_list, gt_labels_list, args.dn_noise_scale)
                         if args.denoising else None)
                out = _model(images, return_riemannian=True, denoising=dn_in)
                # enc_out is always the last element (None when one-stage).
                if args.denoising:
                    boxes_eu, logits, o, R, boxes_ri, dn_out, enc_out = out
                else:
                    boxes_eu, logits, o, R, boxes_ri, enc_out = out
                    dn_out = None

                losses = match_and_compute_loss(
                    boxes_ri, logits,
                    gt_boxes_list, gt_labels_list,
                    o, R, matcher, NUM_CLASSES,
                )
                total = losses["loss_total"]

                # DINO-style denoising auxiliary loss (dn_out is None when the
                # batch has no GT boxes even with --denoising on).
                if dn_out is not None:
                    dn = compute_denoising_loss(
                        dn_out, gt_boxes_list, gt_labels_list,
                        o, R, NUM_CLASSES,
                    )
                    total = total + args.dn_weight * dn["loss_dn"]
                    losses["n_dn"] = dn["n_dn"]

                # Two-stage encoder auxiliary loss — supervises the proposal
                # heads that drive query selection (enc_out is None one-stage).
                if enc_out is not None:
                    enc = compute_encoder_aux_loss(
                        enc_out[0], enc_out[1],
                        gt_boxes_list, gt_labels_list, o, R,
                    )
                    total = total + args.enc_weight * enc["loss_enc"]
                    losses["n_enc"] = enc["n_enc"]

                # scale loss for gradient accumulation
                loss = total / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                set_lr(global_step)
                scaler.step(optimizer)
                scaler.update()
            global_step += 1

            epoch_loss += losses["loss_total"].item()

            if is_main and step % 50 == 0:
                dn_str  = f"  dn={losses['n_dn']}"  if "n_dn"  in losses else ""
                enc_str = f"  enc={losses['n_enc']}" if "n_enc" in losses else ""
                print(f"  epoch {epoch:3d}  step {step:5d}/{len(train_loader)}  "
                      f"loss={losses['loss_total'].item():.4f}  "
                      f"matched={losses['n_matched']}{dn_str}{enc_str}")

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        if is_main:
            metrics = _run_validation(
                model if world_size == 1 else model.module,
                val_loader, device, NUM_CLASSES, CLASS_NAMES
            )
            metrics["loss"] = avg_loss
            elapsed = time.time() - t0
            print(f"epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"mAP={metrics.get('mAP', 0):.4f}  "
                  f"mAP50={metrics.get('mAP_50', 0):.4f}  "
                  f"time={elapsed:.0f}s")
            # Per-distance-bin mAP (paper's headline metric), when available
            dist_bins = [f"{name}={metrics[f'mAP_{name}']:.3f}"
                         for name, _, _ in DISTANCE_BINS
                         if f"mAP_{name}" in metrics]
            if dist_bins:
                print("           mAP by distance(m):  " + "  ".join(dist_bins))
            ckpt_mgr.save(
                model if world_size == 1 else model.module,
                optimizer, epoch, metrics, scaler
            )

    if world_size > 1:
        dist.destroy_process_group()


@torch.no_grad()
def _run_validation(model, val_loader, device, num_classes, class_names) -> dict:
    model.eval()
    evaluator = CocoEvaluator(num_classes=num_classes - 1,
                               class_names=class_names[:-1])
    for images, targets in val_loader:
        images = images.to(device)
        boxes_eu, logits, o, R = model(images)
        probs  = logits.softmax(-1)[:, :, :-1]
        scores, labels = probs.max(-1)
        for b, target in enumerate(targets):
            keep = scores[b] > 0.05
            evaluator.update(
                [DetectionResult(boxes_eu[b][keep].cpu(),
                                  scores[b][keep].cpu(),
                                  labels[b][keep].cpu(),
                                  target["image_id"])],
                [target]
            )
    return evaluator.summarize()


def _load_sam3_backbone(model, ckpt_path, device):
    """Swap SAM3EncoderStub for the real frozen SAM3.1 vision encoder."""
    from telescope.backbone_sam3 import SAM3Backbone
    print(f"[backbone] loading real SAM3.1 vision encoder from {ckpt_path} ...")
    real = SAM3Backbone(
        checkpoint_path=ckpt_path,
        out_channels=model.backbone.out_channels,   # = query_dim (must be 256)
    ).to(device)
    model.backbone = real
    n = sum(p.numel() for p in real.parameters())
    print(f"[backbone] SAM3.1 vision encoder wired ({n/1e6:.0f}M params, will be frozen)")


def _load_efficienttam_backbone(model, ckpt_path, device, config_file):
    """Swap SAM3EncoderStub for the frozen EfficientTAM ViT image encoder."""
    from telescope.backbone_efficienttam import EfficientTAMBackbone
    print(f"[backbone] loading EfficientTAM image encoder from {ckpt_path} "
          f"(config: {config_file}) ...")
    real = EfficientTAMBackbone(
        checkpoint_path=ckpt_path,
        out_channels=model.backbone.out_channels,   # = query_dim (must be 256)
        config_file=config_file,
    ).to(device)
    model.backbone = real
    n = sum(p.numel() for p in real.parameters())
    print(f"[backbone] EfficientTAM image encoder wired ({n/1e6:.0f}M params, "
          f"will be frozen)")


def _oom_hint(args):
    """Print an actionable message when CUDA runs out of memory.

    The defaults (--batch_size 4 --image_size 1024) are paper-scale and assume a
    12-24 GB GPU; on a smaller card the raw OutOfMemoryError traceback is unhelpful.
    """
    free = total = None
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free, total = free_b / 1e9, total_b / 1e9
        except Exception:
            pass
    h, w = args.image_size
    bar = "=" * 72
    print(f"\n{bar}\nCUDA out of memory.")
    print(f"  Current run : --batch_size {args.batch_size}  --image_size {h} {w}"
          + ("  --backbone_ckpt (real SAM3.1)" if args.backbone_ckpt else "  (stub backbone)"))
    if free is not None:
        print(f"  GPU memory  : {free:.1f} GB free of {total:.1f} GB")
    print("\n  The defaults (batch 4 @ 1024) are paper-scale and need a 12-24 GB GPU.")
    print("  On a smaller GPU, try:")
    print("    - --batch_size 1 --image_size 256 256    (smoke test, ~3 GB)")
    print("    - --grad_accum 4                         (effective batch 4, same memory)")
    print("    - close other GPU apps   (check with: nvidia-smi)")
    if args.backbone_ckpt:
        print("    - the real SAM 3.1 backbone needs ~12 GB+ no matter the batch/size;")
        print("      drop --backbone_ckpt to smoke-test with the stub, or use a bigger GPU.")
    print(bar)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        _oom_hint(parse_args())
        raise SystemExit(1)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _oom_hint(parse_args())
            raise SystemExit(1)
        raise
