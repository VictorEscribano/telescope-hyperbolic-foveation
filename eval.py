"""
eval.py — Telescope evaluation script
======================================

Usage:
    # Validation
    python eval.py --data_dir ./data/argoverse2/sensor/val \
                   --checkpoint ./runs/run_01/checkpoint_best.pt

    # Test
    python eval.py --data_dir ./data/argoverse2/sensor/test \
                   --checkpoint ./runs/run_01/checkpoint_best.pt \
                   --split test
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from telescope.pipeline import TelescopeModel
from telescope.data import collate_fn
from telescope.eval import CocoEvaluator, DetectionResult
from telescope.checkpoint import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       type=str, default="argoverse2",
                   choices=["argoverse2", "drones"],
                   help="for drones, point --data_dir at the dataset root")
    p.add_argument("--data_dir",      type=str, required=True)
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--split",         type=str, default="val",
                   choices=["val", "test"])
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--image_size",    type=int, nargs=2, default=[1024, 1024])
    p.add_argument("--score_threshold", type=float, default=0.05)
    p.add_argument("--fp16",          action="store_true", default=True)
    p.add_argument("--backbone", type=str, default="sam3",
                   choices=["sam3", "efficienttam"],
                   help="must match the backbone the checkpoint was trained with")
    p.add_argument("--backbone_ckpt", type=str, default=None,
                   help="backbone checkpoint — required if the model was trained with "
                        "the real backbone, so its weights match at load time")
    p.add_argument("--et_config", type=str,
                   default="configs/efficienttam/efficienttam_s.yaml",
                   help="EfficientTAM Hydra config variant (must match training)")
    p.add_argument("--two_stage",    dest="two_stage", action="store_true", default=True,
                   help="build the DINO two-stage head (default; must match training)")
    p.add_argument("--no_two_stage", dest="two_stage", action="store_false",
                   help="build the one-stage head — use for checkpoints trained with --no_two_stage")
    p.add_argument("--output_file",   type=str, default=None,
                   help="save results JSON to this path")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, score_threshold, fp16, num_classes, class_names):
    model.eval()
    evaluator = CocoEvaluator(
        num_classes = num_classes - 1,
        class_names = class_names[:-1],
    )

    # Foveation diagnostics — see train.py:_run_validation.  Confirms whether the
    # learned lens (o,R) tracks the targets or has collapsed to a fixed point.
    fov_o, fov_R         = [], []
    fov_d_fov, fov_d_cen = [], []
    for images, targets in loader:
        images = images.to(device)
        with autocast(enabled=fp16):
            boxes_eu, logits, o, R = model(images)

        probs  = logits.softmax(-1)[:, :, :-1]   # exclude background
        scores, labels = probs.max(-1)

        fov_o.append(o.detach().float().cpu())
        fov_R.append(R.detach().float().cpu())
        for b, target in enumerate(targets):
            gb = target["boxes"]
            if gb.numel():
                gt_c = gb[:, :2].mean(0).float()              # mean GT centre [-1,1]
                fov_d_fov.append((o[b, :2].detach().float().cpu() - gt_c).norm().item())
                fov_d_cen.append(gt_c.norm().item())
            keep = scores[b] > score_threshold
            evaluator.update(
                [DetectionResult(
                    boxes    = boxes_eu[b][keep].cpu(),
                    scores   = scores[b][keep].cpu(),
                    labels   = labels[b][keep].cpu(),
                    image_id = target["image_id"],
                )],
                [target],
            )

    metrics = evaluator.summarize()
    if fov_o:
        o_all, R_all = torch.cat(fov_o), torch.cat(fov_R)
        metrics["fov/o_x_mean"] = o_all[:, 0].mean().item()
        metrics["fov/o_y_mean"] = o_all[:, 1].mean().item()
        metrics["fov/o_x_std"]  = o_all[:, 0].std(unbiased=False).item()
        metrics["fov/o_y_std"]  = o_all[:, 1].std(unbiased=False).item()
        metrics["fov/R_mean"]   = R_all.mean().item()
        metrics["fov/R_std"]    = R_all.std(unbiased=False).item()
        if fov_d_fov:
            metrics["fov/dist_to_gt"]  = sum(fov_d_fov) / len(fov_d_fov)
            metrics["fov/dist_cen_gt"] = sum(fov_d_cen) / len(fov_d_cen)
    return metrics


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset + class set selection ─────────────────────────────────────────
    if args.dataset == "drones":
        from telescope.data_drones import (DronesYoloDataset as DatasetCls,
                                            DRONE_NUM_CLASSES as NUM_CLASSES,
                                            DRONE_CLASS_NAMES as CLASS_NAMES)
    else:
        from telescope.data import (Argoverse2Dataset as DatasetCls,
                                     NUM_CLASSES, CLASS_NAMES)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TelescopeModel(num_classes=NUM_CLASSES, two_stage=args.two_stage).to(device)

    # Match the training-time architecture so the checkpoint loads: if trained
    # with the real SAM3 backbone, rebuild it before load_state_dict (its weights
    # are then overwritten by the checkpoint's).
    if args.backbone_ckpt:
        if args.backbone == "efficienttam":
            from telescope.backbone_efficienttam import EfficientTAMBackbone
            model.backbone = EfficientTAMBackbone(
                checkpoint_path=args.backbone_ckpt,
                out_channels=model.backbone.out_channels,
                config_file=args.et_config,
            ).to(device)
        else:
            from telescope.backbone_sam3 import SAM3Backbone
            model.backbone = SAM3Backbone(
                checkpoint_path=args.backbone_ckpt,
                out_channels=model.backbone.out_channels,
            ).to(device)

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}  "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = DatasetCls(
        args.data_dir, split=args.split,
        image_size=tuple(args.image_size),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    print(f"Dataset: {len(ds)} frames  ({args.split} split)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(model, loader, device, args.score_threshold, args.fp16,
                       NUM_CLASSES, CLASS_NAMES)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = args.output_file or str(
        Path(args.checkpoint).parent / f"{args.split}_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
