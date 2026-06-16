"""
telescope.eval
==============
COCO-style evaluation utilities for Telescope.

Wraps pycocotools to compute:
    - mAP (COCO IoU 0.5:0.95)
    - mAP_50 (PASCAL IoU 0.5)
    - Per-distance-bin mAP: 0–50m, 50–150m, 150–250m, ≥250m  (TruckDrive protocol)

Requires: pip install pycocotools

Usage::

    evaluator = CocoEvaluator(num_classes=6)

    for images, targets in val_loader:
        preds = model(images)
        evaluator.update(preds, targets)

    metrics = evaluator.summarize()
    print(metrics)
"""

import torch
from torch import Tensor
from typing import List, Dict

__all__ = ["CocoEvaluator", "DetectionResult", "DISTANCE_BINS"]

# Per-distance-bin mAP ranges (metres) — TruckDrive / paper protocol.
# (name, lo, hi); hi is exclusive.  Argoverse2 filters beyond ~300 m.
DISTANCE_BINS = [
    ("0_50",    0.0,   50.0),
    ("50_150",  50.0,  150.0),
    ("150_250", 150.0, 250.0),
    ("250+",    250.0, 1e9),
]


try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


class DetectionResult:
    """Container for one image's detection outputs."""
    __slots__ = ["boxes", "scores", "labels", "image_id"]

    def __init__(
        self,
        boxes:    Tensor,   # (K, 4) Euclidean [cx, cy, w, h]
        scores:   Tensor,   # (K,)
        labels:   Tensor,   # (K,)  integer class indices
        image_id: int,
    ):
        self.boxes    = boxes
        self.scores   = scores
        self.labels   = labels
        self.image_id = image_id


class CocoEvaluator:
    """Accumulate predictions and compute COCO mAP.

    If pycocotools is not installed, falls back to a simple IoU-based
    per-class AP@0.5 approximation.

    Args:
        num_classes  : number of foreground classes (background NOT counted)
        class_names  : optional list of class name strings for pretty printing
    """

    def __init__(
        self,
        num_classes: int = 6,
        class_names: List[str] = None,
    ):
        if not HAS_PYCOCOTOOLS:
            print(
                "WARNING: pycocotools not installed — using simplified AP@0.5.\n"
                "For full COCO metrics: pip install pycocotools"
            )
        self.num_classes  = num_classes
        self.class_names  = class_names or [str(i) for i in range(num_classes)]
        self._predictions: List[Dict] = []
        self._ground_truth: List[Dict] = []
        self._image_ids: List[int] = []

    def update(
        self,
        results:  List[DetectionResult],
        targets:  List[Dict],
    ) -> None:
        """Add one batch of predictions and GT to the accumulator.

        Args:
            results : list of DetectionResult (one per image)
            targets : list of dicts with keys 'boxes' (M,4), 'labels' (M,),
                      'image_id' (int), optionally 'distances' (M,) for
                      distance-binned mAP
        """
        for result, target in zip(results, targets):
            image_id = result.image_id
            self._image_ids.append(image_id)

            boxes_cxcywh = result.boxes.cpu()
            # Convert [cx,cy,w,h] → [x1,y1,w,h] for COCO format
            boxes_xywh = torch.stack([
                boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
                boxes_cxcywh[:, 2],
                boxes_cxcywh[:, 3],
            ], dim=-1)

            for k in range(len(result.scores)):
                self._predictions.append({
                    "image_id":    image_id,
                    "category_id": result.labels[k].item() + 1,  # COCO is 1-indexed
                    "bbox":        boxes_xywh[k].tolist(),
                    "score":       result.scores[k].item(),
                })

            # GT annotations
            gt_boxes = target["boxes"].cpu()
            gt_xywh  = torch.stack([
                gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
                gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
                gt_boxes[:, 2],
                gt_boxes[:, 3],
            ], dim=-1)

            gt_dists = target.get("distances", None)   # (M,) metres, optional

            for k in range(len(target["labels"])):
                self._ground_truth.append({
                    "id":          len(self._ground_truth) + 1,
                    "image_id":    image_id,
                    "category_id": target["labels"][k].item() + 1,
                    "bbox":        gt_xywh[k].tolist(),
                    "area":        (gt_boxes[k, 2] * gt_boxes[k, 3]).item(),
                    "iscrowd":     0,
                    "distance":    (float(gt_dists[k]) if gt_dists is not None
                                    and k < len(gt_dists) else -1.0),
                })

    def summarize(self) -> Dict[str, float]:
        """Compute and return all metrics.

        Returns dict with keys: mAP, mAP_50, mAP_75, and per-class AP.
        """
        if not self._predictions:
            return {"mAP": 0.0, "mAP_50": 0.0}

        if not HAS_PYCOCOTOOLS:
            return self._simple_ap()

        import numpy as np

        # Build COCO-format dataset dicts
        categories = [
            {"id": i + 1, "name": self.class_names[i]}
            for i in range(self.num_classes)
        ]
        images = [{"id": iid} for iid in set(self._image_ids)]

        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": self._ground_truth,
            "categories": categories,
        }
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self._predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        stats = evaluator.stats
        metrics = {
            "mAP":        float(stats[0]),   # IoU 0.50:0.95, all sizes
            "mAP_50":     float(stats[1]),   # IoU 0.50 (PASCAL)
            "mAP_75":     float(stats[2]),   # IoU 0.75
            "mAP_small":  float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large":  float(stats[5]),
        }

        # ── Per-distance-bin mAP (TruckDrive protocol) ────────────────────────
        # Only computed when GT carry real distances (Argoverse2Dataset provides
        # them; synthetic targets default to -1 and are skipped).
        has_dist = any(g.get("distance", -1.0) >= 0 for g in self._ground_truth)
        if has_dist:
            for name, lo, hi in DISTANCE_BINS:
                res = self._eval_distance_bin(categories, images, lo, hi)
                if res is not None:
                    metrics[f"mAP_{name}"]    = res[0]
                    metrics[f"mAP_50_{name}"] = res[1]
        return metrics

    def _eval_distance_bin(self, categories, images, lo: float, hi: float):
        """COCO mAP / mAP@50 restricted to GT in distance range [lo, hi) metres.

        GT outside the range are flagged ``ignore=1`` so COCOeval neither counts
        them as misses nor penalises detections that match them — the standard
        way to slice mAP by a per-GT attribute.  Returns (mAP, mAP_50) or None
        if the bin has no GT.
        """
        import copy
        anns = copy.deepcopy(self._ground_truth)
        n_in = 0
        for a in anns:
            d = a.get("distance", -1.0)
            in_bin = (d >= 0) and (lo <= d < hi)
            a["ignore"] = 0 if in_bin else 1
            n_in += int(in_bin)
        if n_in == 0:
            return None

        coco_gt = COCO()
        coco_gt.dataset = {"info": {}, "licenses": [], "images": images,
                           "annotations": anns, "categories": categories}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(self._predictions)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return float(ev.stats[0]), float(ev.stats[1])

    def reset(self) -> None:
        """Clear accumulated predictions — call between epochs."""
        self._predictions.clear()
        self._ground_truth.clear()
        self._image_ids.clear()

    def _simple_ap(self) -> Dict[str, float]:
        """Fallback: per-class AP@0.5 without pycocotools."""
        # Group predictions by class
        per_class = {c: [] for c in range(self.num_classes)}
        for pred in self._predictions:
            c = pred["category_id"] - 1
            per_class[c].append(pred)

        per_class_gt = {c: [] for c in range(self.num_classes)}
        for ann in self._ground_truth:
            c = ann["category_id"] - 1
            per_class_gt[c].append(ann)

        aps = []
        for c in range(self.num_classes):
            preds = sorted(per_class[c], key=lambda x: -x["score"])
            gts   = per_class_gt[c]
            if not gts:
                continue

            tp = torch.zeros(len(preds))
            fp = torch.zeros(len(preds))
            matched = set()

            for i, pred in enumerate(preds):
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gts):
                    if j in matched:
                        continue
                    if gt["image_id"] != pred["image_id"]:
                        continue
                    iou = _box_iou_xywh(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.5:
                    tp[i] = 1
                    matched.add(best_j)
                else:
                    fp[i] = 1

            tp_cum = tp.cumsum(0)
            fp_cum = fp.cumsum(0)
            recall    = tp_cum / max(len(gts), 1)
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)
            aps.append(_voc_ap(recall.numpy(), precision.numpy()))

        mAP = float(sum(aps) / len(aps)) if aps else 0.0
        return {"mAP_50": mAP, "mAP": mAP}


def _box_iou_xywh(b1, b2):
    """IoU between two [x, y, w, h] boxes."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def _voc_ap(recall, precision):
    """Compute VOC-style AP from recall/precision arrays."""
    import numpy as np
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
