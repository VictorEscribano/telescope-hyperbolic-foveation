"""
telescope.checkpoint
====================
Checkpoint save / load with:
    - Best-model tracking (by mAP)
    - Last-N rotation (avoids filling disk)
    - Resume-from-checkpoint (epoch, optimizer state, scaler state)
"""

import os
import re
import torch
from pathlib import Path
from typing import Optional

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """Save and load training checkpoints.

    Usage::

        ckpt = CheckpointManager(save_dir='checkpoints/run_01', keep_last=3)

        # At the end of each epoch:
        ckpt.save(model, optimizer, scaler, epoch=epoch, metrics={'mAP': 0.32})

        # To resume:
        start_epoch = ckpt.load_latest(model, optimizer, scaler)

    Args:
        save_dir  : directory where checkpoints are stored
        keep_last : number of most-recent checkpoints to keep (oldest deleted)
        metric_key: metric used to track best model (higher = better)
    """

    def __init__(
        self,
        save_dir:   str,
        keep_last:  int = 3,
        metric_key: str = "mAP",
    ) -> None:
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last  = keep_last
        self.metric_key = metric_key
        self._best_metric = -float("inf")

    def save(
        self,
        model:     torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch:     int,
        metrics:   dict,
        scaler:    Optional[object] = None,   # torch.cuda.amp.GradScaler
    ) -> Path:
        """Save a checkpoint and maintain the rotation window.

        Returns the path of the saved file.
        """
        state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":              metrics,
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        path = self.save_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(state, path)
        print(f"[checkpoint] saved  → {path}  ({self.metric_key}={metrics.get(self.metric_key, 'n/a')})")

        # Track best model separately
        metric_val = metrics.get(self.metric_key, -float("inf"))
        if metric_val > self._best_metric:
            self._best_metric = metric_val
            best_path = self.save_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            print(f"[checkpoint] new best → {best_path}  ({self.metric_key}={metric_val:.4f})")

        # Rotation: delete oldest if over keep_last
        self._rotate()

        return path

    def load_latest(
        self,
        model:     torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler:    Optional[object] = None,
        device:    str = "cpu",
    ) -> int:
        """Load the most recent checkpoint.

        Returns the next epoch to start from (0 if no checkpoint found).
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            print("[checkpoint] no checkpoint found, starting from epoch 0")
            return 0
        return self._load(checkpoints[-1], model, optimizer, scaler, device)

    def load_best(
        self,
        model:     torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler:    Optional[object] = None,
        device:    str = "cpu",
    ) -> int:
        """Load the best checkpoint (by metric_key)."""
        best = self.save_dir / "checkpoint_best.pt"
        if not best.exists():
            print("[checkpoint] no best checkpoint found")
            return 0
        return self._load(best, model, optimizer, scaler, device)

    def _load(self, path, model, optimizer, scaler, device) -> int:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        epoch = state.get("epoch", 0)
        metrics = state.get("metrics", {})
        print(f"[checkpoint] loaded ← {path}  epoch={epoch}  {metrics}")
        return epoch + 1   # resume from NEXT epoch

    def _list_checkpoints(self):
        """Return sorted list of checkpoint_epochXXXX.pt paths."""
        pattern = re.compile(r"checkpoint_epoch(\d+)\.pt$")
        ckpts = sorted(
            (p for p in self.save_dir.glob("checkpoint_epoch*.pt")
             if pattern.search(p.name)),
            key=lambda p: int(pattern.search(p.name).group(1)),
        )
        return ckpts

    def _rotate(self):
        """Delete oldest checkpoints beyond keep_last."""
        ckpts = self._list_checkpoints()
        for old in ckpts[:-self.keep_last]:
            old.unlink()
            print(f"[checkpoint] deleted  {old.name}")
