"""
compare.py — Compare Telescope vs baseline runs
=================================================

Reads the test_results.json files saved by eval.py from two or more
run directories and produces a side-by-side bar chart and table.

Usage:
    python compare.py \
        --runs ./runs/baseline ./runs/telescope \
        --labels "No foveation" "Telescope"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs",   type=str, nargs="+", required=True,
                   help="paths to run directories containing test_results.json")
    p.add_argument("--labels", type=str, nargs="+", default=None,
                   help="display name for each run (same order as --runs)")
    p.add_argument("--metric_file", type=str, default="test_results.json",
                   help="filename to look for inside each run dir")
    p.add_argument("--output", type=str, default="comparison.png")
    return p.parse_args()


METRICS_TO_PLOT = ["mAP_50", "mAP", "mAP_75"]
# Real per-distance-bin keys produced by telescope.eval (TruckDrive protocol).
# Missing keys fall back to 0 in _bar_panel, so this is safe on older results.
DISTANCE_METRICS = {
    "0–50 m":    "mAP_0_50",
    "50–150 m":  "mAP_50_150",
    "150–250 m": "mAP_150_250",
    "≥250 m":    "mAP_250+",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_metrics(run_dir: str, metric_file: str) -> dict:
    path = Path(run_dir) / metric_file
    if not path.exists():
        raise FileNotFoundError(
            f"No results file found at {path}.\n"
            f"Run: python eval.py --data_dir ... --checkpoint {run_dir}/checkpoint_best.pt"
        )
    with open(path) as f:
        return json.load(f)


def main():
    args   = parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    assert len(labels) == len(args.runs), "--labels must have same length as --runs"

    all_metrics = []
    for run, label in zip(args.runs, labels):
        m = load_metrics(run, args.metric_file)
        all_metrics.append((label, m))
        print(f"\n{label}:")
        for k, v in m.items():
            print(f"  {k:<15}: {v:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: overall metrics
    _bar_panel(
        ax        = axes[0],
        all_metrics = all_metrics,
        keys      = METRICS_TO_PLOT,
        title     = "Overall detection metrics",
    )

    # Panel 2: distance-binned metrics
    _bar_panel(
        ax          = axes[1],
        all_metrics = all_metrics,
        keys        = list(DISTANCE_METRICS.values()),
        xlabels     = list(DISTANCE_METRICS.keys()),
        title       = "Detection by distance range",
    )

    plt.suptitle("Telescope vs baseline comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison chart → {args.output}")
    plt.show()


def _bar_panel(ax, all_metrics, keys, title, xlabels=None):
    n_groups = len(keys)
    n_runs   = len(all_metrics)
    w        = 0.7 / n_runs
    x        = np.arange(n_groups)

    for i, (label, metrics) in enumerate(all_metrics):
        vals = [metrics.get(k, 0.0) for k in keys]
        bars = ax.bar(
            x + (i - n_runs / 2 + 0.5) * w, vals,
            width=w, label=label, color=COLORS[i % len(COLORS)], alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels or keys, fontsize=9)
    ax.set_ylabel("mAP")
    ax.set_ylim(0, min(1.0, max(
        metrics.get(k, 0) for _, metrics in all_metrics for k in keys
    ) * 1.25 + 0.05))
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


if __name__ == "__main__":
    main()
