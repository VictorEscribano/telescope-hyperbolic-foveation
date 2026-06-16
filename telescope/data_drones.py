"""
telescope.data_drones
=====================
PyTorch Dataset for a YOLO-format drone-detection dataset, as a 2-D
(no-LiDAR) target for Telescope.

Layout expected (Ultralytics YOLO):
    root/
      data.yaml                 (names: ['Drone'], nc: 1)
      train/images/*.jpg        train/labels/*.txt
      val/images/*.jpg          val/labels/*.txt
      test/images/*.jpg         test/labels/*.txt

Each label file holds one box per line, ``class cx cy w h`` normalised to
[0, 1].  Images **without** a label file are treated as pure-background
negatives (no boxes) — about half of this dataset.

Why YOLO over the COCO/RF-DETR copy: the boxes are already normalised, so
the loader is robust to the dataset's mixed image sizes (640×480 … 1920×1080)
without reading per-image dimensions, and it has the larger train split.

Boxes are returned as ``[cx, cy, w, h]`` in **[-1, 1]** (Telescope's convention),
matching :class:`telescope.data.Argoverse2Dataset` so train.py/eval.py can use
either via a ``--dataset`` switch.  There is no 3-D range here, so ``distances``
is empty and the per-distance-bin mAP is skipped by the evaluator.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .data import collate_fn   # identical collate (images stacked, targets list)

__all__ = ["DronesYoloDataset", "collate_fn", "DRONE_CLASS_NAMES", "DRONE_NUM_CLASSES"]

# Single foreground class + background (last index), mirroring telescope.data.
DRONE_CLASS_NAMES = ["Drone", "__background__"]
DRONE_NUM_CLASSES = len(DRONE_CLASS_NAMES)          # = 2

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# YOLO uses 'val'; accept 'valid' (RF-DETR/Roboflow) as an alias.
_SPLIT_ALIASES = {"valid": "val"}


class DronesYoloDataset(Dataset):
    """YOLO-format drone dataset → Telescope targets.

    Args:
        root_dir   : dataset root containing ``<split>/images`` and ``<split>/labels``
        split      : 'train', 'val', or 'test'
        image_size : (H, W) to resize images to — default (1024, 1024)
        keep_empty : keep images that have no label file (background negatives).
                     True (default) trains on negatives too; set False to use
                     only images that contain at least one drone.
    """

    def __init__(
        self,
        root_dir:   str,
        split:      str = "train",
        image_size: Tuple[int, int] = (1024, 1024),
        keep_empty: bool = True,
    ) -> None:
        split = _SPLIT_ALIASES.get(split, split)
        self.root       = Path(root_dir)
        self.split      = split
        self.image_size = image_size
        self.img_dir    = self.root / split / "images"
        self.lbl_dir    = self.root / split / "labels"

        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"{self.img_dir} not found. Expected <root>/<split>/images "
                f"(root={self.root}, split={split})."
            )

        images = sorted(p for p in self.img_dir.iterdir()
                        if p.suffix.lower() in _IMG_EXTS)
        if not keep_empty:
            images = [p for p in images if self._label_path(p).is_file()]
        if not images:
            raise RuntimeError(f"No images found under {self.img_dir}")
        self.images: List[Path] = images

    def _label_path(self, img_path: Path) -> Path:
        return self.lbl_dir / f"{img_path.stem}.txt"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:
        img_path = self.images[idx]
        image    = self._load_image(img_path)              # (3, H0, W0) in [0,1]
        image    = F.interpolate(
            image.unsqueeze(0), size=self.image_size,
            mode="bilinear", align_corners=True,
        ).squeeze(0)                                        # (3, H, W)

        boxes, labels = self._load_label(self._label_path(img_path))

        return image, {
            "boxes":     boxes,                             # (M,4) [cx,cy,w,h] in [-1,1]
            "labels":    labels,                            # (M,)  all 0 (Drone)
            "distances": torch.full((len(labels),), -1.0, dtype=torch.float32),  # no 3-D range → -1 = unknown, so the evaluator skips per-distance-bin mAP
            "image_id":  idx,
            "file_name": img_path.name,
        }

    def _load_image(self, img_path: Path) -> Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(img_path).convert("RGB")
        return TF.to_tensor(img)                            # (3, H, W) in [0,1]

    def _load_label(self, lbl_path: Path) -> Tuple[Tensor, Tensor]:
        """Parse a YOLO label file → boxes in [-1,1], labels.  Missing → empty."""
        if not lbl_path.is_file():
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

        boxes, labels = [], []
        try:
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = (float(x) for x in parts[:5])
                if w <= 0 or h <= 0:
                    continue
                # YOLO [0,1] → Telescope [-1,1]: centre c*2-1, size spans 2 units → *2.
                boxes.append([cx * 2 - 1, cy * 2 - 1, w * 2, h * 2])
                labels.append(int(cls))                     # single class → 0
        except Exception as exc:                            # noqa: BLE001
            warnings.warn(f"[DronesYoloDataset] bad label {lbl_path}: {exc}")
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

        if not boxes:
            return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
        return (torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long))


# ── Quick self-test (run on the machine that has the dataset) ──────────────────
#   python -m telescope.data_drones --root /home/ia/Documentos/YOLO/datasets/drones_v5
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n",     type=int, default=2000, help="images to scan for box stats")
    args = ap.parse_args()

    ds = DronesYoloDataset(args.root, split=args.split)
    print(f"split={args.split}  images={len(ds)}  classes={DRONE_CLASS_NAMES}")

    img, tgt = ds[0]
    print(f"image tensor : {tuple(img.shape)}  range[{img.min():.2f},{img.max():.2f}]")
    print(f"sample target: boxes={tuple(tgt['boxes'].shape)}  labels={tgt['labels'].tolist()[:5]}")
    if len(tgt["boxes"]):
        print(f"  box[0] (cx,cy,w,h in [-1,1]) = {[round(v,3) for v in tgt['boxes'][0].tolist()]}")

    # Box-size + objects-per-image stats over a sample.
    import random
    n_boxes = n_empty = 0
    ws, hs, per_img = [], [], []
    for i in random.sample(range(len(ds)), min(args.n, len(ds))):
        b = ds._load_label(ds._label_path(ds.images[i]))[0]
        per_img.append(len(b))
        if len(b) == 0:
            n_empty += 1
        else:
            n_boxes += len(b)
            ws += (b[:, 2] / 2).tolist()   # back to [0,1] fraction
            hs += (b[:, 3] / 2).tolist()
    import statistics as st
    scanned = min(args.n, len(ds))
    print(f"\nover {scanned} sampled images:")
    print(f"  empty (background) : {n_empty} ({100*n_empty/scanned:.0f}%)")
    print(f"  boxes total        : {n_boxes}  (avg {n_boxes/max(scanned,1):.2f}/img, "
          f"max {max(per_img)}/img)")
    if ws:
        print(f"  box width  frac    : median {st.median(ws):.3f}  min {min(ws):.4f}  max {max(ws):.3f}")
        print(f"  box height frac    : median {st.median(hs):.3f}  min {min(hs):.4f}  max {max(hs):.3f}")
        print(f"  → at 1024px median ≈ {st.median(ws)*1024:.0f}×{st.median(hs)*1024:.0f} px,"
              f" smallest ≈ {min(ws)*1024:.0f}px wide")
