"""
telescope.trainlog
==================
Ultralytics-style training logger for Telescope.

Produces, in ``output_dir``:
  - ``results.csv`` — one row per epoch (machine-readable, plot-anywhere).
  - a tidy console table (header once, one aligned row per epoch).
  - ``results.png`` — summary curves (losses, mAP, recall, lr) at the end.

Only rank 0 should construct/use this in DDP runs.
"""

import csv
from pathlib import Path

__all__ = ["MetricsLogger"]

# CSV column order (kept stable so results.csv is easy to diff / re-plot).
COLUMNS = [
    "epoch", "time_s", "lr",
    "train/loss", "train/l1", "train/giou", "train/cls",
    "val/loss",
    "metrics/mAP50-95", "metrics/mAP50", "metrics/recall",
]

# (csv key, plot title) for the panels in results.png.
_PANELS = [
    ("train/loss",       "train loss"),
    ("val/loss",         "val loss"),
    ("metrics/mAP50",    "mAP@50"),
    ("metrics/mAP50-95", "mAP@50-95"),
    ("metrics/recall",   "recall (AR@100)"),
    ("lr",               "learning rate"),
]


class MetricsLogger:
    def __init__(self, out_dir, total_epochs: int):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "results.csv"
        self.total = total_epochs
        self.rows = []
        self._header_printed = False
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(COLUMNS)

    def _print_header(self):
        print(f"\n{'Epoch':>9} {'GPU_mem':>8} {'loss':>9} {'val_loss':>9} "
              f"{'mAP50':>8} {'mAP50-95':>9} {'recall':>8} {'time':>7}")
        self._header_printed = True

    def log(self, epoch: int, row: dict, gpu_mem_gb: float = 0.0):
        """Append one epoch to results.csv and print its console row."""
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([_fmt(row.get(c, "")) for c in COLUMNS])
        self.rows.append(row)

        if not self._header_printed:
            self._print_header()
        print(f"{epoch + 1:>4}/{self.total:<4} {gpu_mem_gb:>6.1f}G "
              f"{row['train/loss']:>9.4f} {row['val/loss']:>9.4f} "
              f"{row['metrics/mAP50']:>8.4f} {row['metrics/mAP50-95']:>9.4f} "
              f"{row['metrics/recall']:>8.4f} {row['time_s']:>6.0f}s")

    def plot(self):
        """Render results.png from the accumulated rows (best-effort)."""
        if not self.rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless / air-gapped server
            import matplotlib.pyplot as plt
        except Exception as e:  # matplotlib missing → CSV is still there
            print(f"[plot] matplotlib unavailable ({e}); skipping results.png. "
                  f"Plot from {self.csv_path} on another machine.")
            return

        ep = [r["epoch"] + 1 for r in self.rows]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, (key, title) in zip(axes.ravel(), _PANELS):
            y = [r.get(key, float("nan")) for r in self.rows]
            ax.plot(ep, y, marker=".", linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
        fig.suptitle("Telescope training", fontweight="bold")
        fig.tight_layout()
        out = self.dir / "results.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[plot] saved {out}")


def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v
