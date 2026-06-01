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

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

from telescope.pipeline import TelescopeModel
from telescope.matcher import HungarianMatcher, match_and_compute_loss
from telescope.eval import CocoEvaluator, DetectionResult
from telescope.data import Argoverse2Dataset, collate_fn, NUM_CLASSES, CLASS_NAMES
from telescope.checkpoint import CheckpointManager


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
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
    p.add_argument("--backbone_ckpt", type=str, default=None,
                   help="path to SAM3.1 checkpoint (optional — uses stub if not given)")
    p.add_argument("--no_foveation", action="store_true", default=False,
                   help="disable foveation (R fixed near zero) for ablation baseline")
    p.add_argument("--grad_accum",   type=int, default=1,
                   help="gradient accumulation steps (use 2 on 14GB VRAM for effective batch=4)")
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

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TelescopeModel(
        num_classes = NUM_CLASSES,
        num_queries = args.num_queries,
        query_dim   = args.query_dim,
    ).to(device)

    # Baseline ablation: fix R ≈ 0 so Phi(x) = x everywhere
    if args.no_foveation:
        print("[baseline] foveation disabled — R fixed to 0.001 (identity warp)")
        for param in model.fov_estimator.parameters():
            param.requires_grad_(False)
        model.fov_estimator._no_foveation = True   # checked in forward below

    # Optionally load real SAM3.1 backbone
    if args.backbone_ckpt:
        _load_sam3_backbone(model, args.backbone_ckpt, device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                   weight_decay=args.weight_decay)
    # Lambda LR with 1 warm-up epoch (paper Table 9)
    def lr_lambda(epoch):
        if epoch < 1:
            return epoch + 1e-8          # warm-up: linear ramp
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=args.fp16)

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
    train_ds = Argoverse2Dataset(args.data_dir, split="train",
                                  image_size=tuple(args.image_size))
    val_ds   = Argoverse2Dataset(args.val_dir,  split="val",
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

            with autocast(enabled=args.fp16):
                _model = model.module if world_size > 1 else model

                # baseline ablation: override o,R so Phi = identity
                boxes_eu, logits, o, R, boxes_ri = _model(
                    images, return_riemannian=True
                )
                if args.no_foveation:
                    R = torch.full_like(R, 0.001)   # w(r)=0 → Phi(x)=x

                losses = match_and_compute_loss(
                    boxes_ri, logits,
                    gt_boxes_list, gt_labels_list,
                    o, R, matcher, NUM_CLASSES,
                )
                # scale loss for gradient accumulation
                loss = losses["loss_total"] / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += losses["loss_total"].item()

            if is_main and step % 50 == 0:
                print(f"  epoch {epoch:3d}  step {step:5d}/{len(train_loader)}  "
                      f"loss={losses['loss_total'].item():.4f}  "
                      f"matched={losses['n_matched']}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        if is_main:
            metrics = _run_validation(
                model if world_size == 1 else model.module,
                val_loader, device
            )
            metrics["loss"] = avg_loss
            elapsed = time.time() - t0
            print(f"epoch {epoch:3d}  loss={avg_loss:.4f}  "
                  f"mAP={metrics.get('mAP_50', 0):.4f}  "
                  f"time={elapsed:.0f}s")
            ckpt_mgr.save(
                model if world_size == 1 else model.module,
                optimizer, epoch, metrics, scaler
            )

    if world_size > 1:
        dist.destroy_process_group()


@torch.no_grad()
def _run_validation(model, val_loader, device) -> dict:
    model.eval()
    evaluator = CocoEvaluator(num_classes=NUM_CLASSES - 1,
                               class_names=CLASS_NAMES[:-1])
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
    """Swap SAM3EncoderStub for real SAM3.1 weights once available."""
    # TODO: replace with actual SAM3 loading API when confirmed
    # from sam3 import build_sam3
    # sam3 = build_sam3(config="...", checkpoint=ckpt_path)
    # model.backbone = sam3.image_encoder
    print(f"[backbone] SAM3.1 loading from {ckpt_path} — TODO: wire up after API confirmed")


if __name__ == "__main__":
    main()
