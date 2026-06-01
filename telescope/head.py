"""
telescope.head
==============
Detection head components specific to Telescope.

Standard components (encoder/decoder transformer blocks) are assumed to
come from an external library (mmdetection, detrex, or HuggingFace
transformers).  This module implements only what is UNIQUE to Telescope:

  - RiemannianBoxHead   : MLP that predicts b' in Riemannian space
  - TelescopeLoss       : gIoU + L1 loss computed in Euclidean space
  - denoise_boxes       : helper for the DINO-style denoising training scheme

Reference: Telescope paper §3.2, §4 (Training Losses, Denoising).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .geometry import EPS
from .box import euclidean_to_riemannian_box, riemannian_to_euclidean_box

__all__ = [
    "RiemannianBoxHead",
    "TelescopeLoss",
    "denoise_boxes",
    "generalized_box_iou",
]


# ── Riemannian box prediction head ────────────────────────────────────────────

class RiemannianBoxHead(nn.Module):
    """3-layer MLP that predicts Riemannian boxes from DETR query features.

    Predicts b' = [Phi(cx), Phi(cy), ||t_x||, ||t_y||] in the induced
    Riemannian space.  The centre coordinates are predicted via sigmoid
    (normalised to [0, 1] range, then mapped to [-1, 1]).  The tangent
    norms are predicted via softplus (strictly positive).

    Args:
        query_dim    : Dimension of input DETR query features.  Default 256.
        num_classes  : Number of object classes + 1 (background).
        hidden       : Width of the hidden layers.
    """

    def __init__(
        self,
        query_dim: int = 256,
        num_classes: int = 6,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Box regression head: predicts [cx, cy, w_norm, h_norm]
        self.box_head = nn.Sequential(
            nn.Linear(query_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, 4),
        )

        # Class logits head
        self.class_head = nn.Linear(query_dim, num_classes)

    def forward(self, query_features: Tensor):
        """
        Args:
            query_features : (B, num_queries, query_dim)
        Returns:
            boxes_ri  : (B, num_queries, 4) — [Phi(cx), Phi(cy), ||tx||, ||ty||]
                        cx/cy predicted via sigmoid → mapped to [-1, 1]
                        tx/ty norms predicted via softplus (positive)
            class_logits : (B, num_queries, num_classes)
        """
        raw = self.box_head(query_features)           # (B, Q, 4)

        # Centre: sigmoid → [0, 1] → map to [-1, 1]
        centre = torch.sigmoid(raw[..., :2]) * 2 - 1   # (B, Q, 2) in [-1, 1]

        # Tangent norms: softplus → strictly positive
        tangents = F.softplus(raw[..., 2:])            # (B, Q, 2) > 0

        boxes_ri     = torch.cat([centre, tangents], dim=-1)  # (B, Q, 4)
        class_logits = self.class_head(query_features)         # (B, Q, C)
        return boxes_ri, class_logits


# ── Generalised IoU ───────────────────────────────────────────────────────────

def generalized_box_iou(
    boxes1: Tensor,
    boxes2: Tensor,
) -> Tensor:
    """Generalised Intersection-over-Union for pairs of [cx, cy, w, h] boxes.

    gIoU provides a gradient even when boxes do not overlap — critical for
    training on small distant objects where IoU = 0 is common.

    Both inputs must be in the same coordinate system (Euclidean, after Phi^{-1}).

    Args:
        boxes1 : (N, 4) [cx, cy, w, h]
        boxes2 : (N, 4) [cx, cy, w, h]
    Returns:
        (N,) gIoU values in [-1, 1]
    """
    # Convert [cx, cy, w, h] → [x1, y1, x2, y2]
    def to_xyxy(b):
        return torch.stack([
            b[..., 0] - b[..., 2] / 2,
            b[..., 1] - b[..., 3] / 2,
            b[..., 0] + b[..., 2] / 2,
            b[..., 1] + b[..., 3] / 2,
        ], dim=-1)

    b1 = to_xyxy(boxes1)
    b2 = to_xyxy(boxes2)

    # Intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    # Union
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    union = area1 + area2 - inter + EPS
    iou   = inter / union

    # Smallest enclosing box
    enc_x1 = torch.min(b1[..., 0], b2[..., 0])
    enc_y1 = torch.min(b1[..., 1], b2[..., 1])
    enc_x2 = torch.max(b1[..., 2], b2[..., 2])
    enc_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc    = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + EPS

    return iou - (enc - union) / enc   # (N,) in [-1, 1]


# ── Telescope loss ────────────────────────────────────────────────────────────

class TelescopeLoss(nn.Module):
    """Combined loss for Telescope detection.

    Loss pipeline (paper §4, Training Losses):
        1. Predicted boxes are in Riemannian space (b').
        2. Apply Phi^{-1} to convert both predicted and GT boxes to Euclidean.
        3. Compute L1 + gIoU losses in Euclidean space.
        4. Add a cross-entropy classification loss.

    This ensures the loss gradient flows correctly through Phi^{-1} back
    to the box head and the foveation parameters.

    Args:
        lambda_l1   : Weight for the L1 box regression loss.
        lambda_giou : Weight for the gIoU loss.
        lambda_cls  : Weight for the classification loss.
    """

    def __init__(
        self,
        lambda_l1:   float = 5.0,
        lambda_giou: float = 2.0,
        lambda_cls:  float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_l1   = lambda_l1
        self.lambda_giou = lambda_giou
        self.lambda_cls  = lambda_cls

    def forward(
        self,
        pred_boxes_ri:    Tensor,
        pred_class_logits: Tensor,
        gt_boxes_eu:      Tensor,
        gt_labels:        Tensor,
        o: Tensor,
        R: Tensor,
        alpha: float = 2.0,
        p: float = 2.0,
    ) -> dict:
        """Compute the full Telescope detection loss.

        Args:
            pred_boxes_ri     : (N, 4) predicted Riemannian boxes [Phi(c), ||tx||, ||ty||]
            pred_class_logits : (N, num_classes)
            gt_boxes_eu       : (N, 4) ground-truth Euclidean boxes [cx, cy, w, h]
            gt_labels         : (N,) integer class indices
            o, R              : Foveation parameters for Phi^{-1}
        Returns:
            dict with keys: loss_total, loss_l1, loss_giou, loss_cls
        """
        # Step 1: decode predicted Riemannian boxes → Euclidean
        pred_boxes_eu = riemannian_to_euclidean_box(pred_boxes_ri, o, R, alpha, p)

        # Step 2: box regression losses (Euclidean space)
        loss_l1   = F.l1_loss(pred_boxes_eu, gt_boxes_eu, reduction='mean')
        giou_vals = generalized_box_iou(pred_boxes_eu, gt_boxes_eu)
        loss_giou = (1.0 - giou_vals).mean()

        # Step 3: classification loss
        loss_cls = F.cross_entropy(pred_class_logits, gt_labels)

        loss_total = (
            self.lambda_l1   * loss_l1
            + self.lambda_giou * loss_giou
            + self.lambda_cls  * loss_cls
        )

        return dict(
            loss_total=loss_total,
            loss_l1=loss_l1,
            loss_giou=loss_giou,
            loss_cls=loss_cls,
        )


# ── Denoising helper ──────────────────────────────────────────────────────────

def denoise_boxes(
    gt_boxes_eu: Tensor,
    o: Tensor,
    R: Tensor,
    noise_scale: float = 0.4,
    alpha: float = 2.0,
    p: float = 2.0,
) -> Tensor:
    """Add noise to GT boxes and project them into Riemannian space.

    Implements the DINO-style denoising training scheme (paper §4):
        1. Add noise to GT Euclidean boxes.
        2. Project to Riemannian space via euclidean_to_riemannian_box.
        3. These noisy Riemannian boxes are concatenated to the object
           queries so the decoder learns to de-noise them.

    Args:
        gt_boxes_eu : (N, 4) ground-truth Euclidean boxes
        noise_scale : std of Gaussian noise relative to box size
    Returns:
        (N, 4) noisy Riemannian boxes, ready to be appended to queries
    """
    noise = torch.randn_like(gt_boxes_eu) * noise_scale
    # Scale noise by box dimensions (w, h) so it is proportional
    scale = torch.cat([
        gt_boxes_eu[:, 2:3],   # w
        gt_boxes_eu[:, 3:4],   # h
        gt_boxes_eu[:, 2:3],
        gt_boxes_eu[:, 3:4],
    ], dim=-1)
    noisy_eu = gt_boxes_eu + noise * scale
    return euclidean_to_riemannian_box(noisy_eu, o, R, alpha, p)
