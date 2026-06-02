"""
telescope.matcher
=================
Hungarian matcher for DETR-style training.

During training, DETR predicts N queries but the image has M < N GT boxes.
The Hungarian algorithm finds the optimal 1-to-1 assignment between predictions
and GT boxes by minimising a combined cost (class + L1 + gIoU).

Un-matched queries are assigned the "no-object" class and receive no box loss.

Reference: Carion et al. "End-to-End Object Detection with Transformers" (DETR, 2020).
The Telescope adaptation: boxes are decoded from Riemannian space via Phi^{-1}
before the matching cost is computed, so the matcher always operates in Euclidean space.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from .box import riemannian_to_euclidean_box
from .head import generalized_box_iou

__all__ = ["HungarianMatcher", "match_and_compute_loss", "compute_denoising_loss"]


class HungarianMatcher:
    """Compute the optimal assignment between DETR predictions and GT boxes.

    Cost matrix per image:
        C = λ_cls · C_cls  +  λ_L1 · C_L1  +  λ_gIoU · C_gIoU

    All costs are computed in Euclidean space after decoding predicted
    Riemannian boxes via Phi^{-1}.

    Args:
        cost_cls  : weight for classification cost
        cost_l1   : weight for L1 box cost
        cost_giou : weight for gIoU cost
    """

    def __init__(
        self,
        cost_cls:  float = 1.0,
        cost_l1:   float = 5.0,
        cost_giou: float = 2.0,
    ) -> None:
        self.cost_cls  = cost_cls
        self.cost_l1   = cost_l1
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(
        self,
        pred_boxes_ri:    Tensor,   # (Q, 4) Riemannian predictions for ONE image
        pred_logits:      Tensor,   # (Q, num_classes)
        gt_boxes_eu:      Tensor,   # (M, 4) GT boxes [cx, cy, w, h]
        gt_labels:        Tensor,   # (M,)   integer class indices
        o:                Tensor,   # (2,) foveation centre for this image
        R:                Tensor,   # scalar foveation radius
        alpha:            float = 2.0,
        p:                float = 2.0,
    ):
        """
        Returns:
            pred_idx : (K,) indices of matched predictions  (K = min(Q, M))
            gt_idx   : (K,) corresponding GT indices
        """
        Q = pred_boxes_ri.shape[0]
        M = gt_boxes_eu.shape[0]

        if M == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        # ── Decode predicted boxes to Euclidean space ─────────────────────────
        pred_boxes_eu = riemannian_to_euclidean_box(pred_boxes_ri, o, R, alpha, p)  # (Q, 4)

        # ── Classification cost: negative probability of the correct class ────
        # pred_logits: (Q, C),  gt_labels: (M,)
        # cost_cls[q, m] = -softmax(pred_logits)[q, gt_labels[m]]
        pred_probs = pred_logits.softmax(-1)                        # (Q, C)
        cost_cls   = -pred_probs[:, gt_labels]                      # (Q, M)

        # ── L1 box cost ───────────────────────────────────────────────────────
        # cost_l1[q, m] = L1(pred_eu[q], gt_eu[m])
        # broadcast: pred (Q,4) vs gt (M,4)
        cost_l1 = torch.cdist(pred_boxes_eu, gt_boxes_eu, p=1)      # (Q, M)

        # ── gIoU cost ─────────────────────────────────────────────────────────
        # Expand to all (q, m) pairs
        pred_exp = pred_boxes_eu.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)  # (Q*M, 4)
        gt_exp   = gt_boxes_eu.unsqueeze(0).expand(Q, -1, -1).reshape(-1, 4)   # (Q*M, 4)
        giou     = generalized_box_iou(pred_exp, gt_exp).reshape(Q, M)          # (Q, M)
        cost_giou = -giou                                                         # (Q, M)

        # ── Combined cost ─────────────────────────────────────────────────────
        C = (
            self.cost_cls  * cost_cls
            + self.cost_l1   * cost_l1
            + self.cost_giou * cost_giou
        ).cpu().numpy()

        # ── Hungarian algorithm (scipy) ───────────────────────────────────────
        pred_idx, gt_idx = linear_sum_assignment(C)

        return (
            torch.as_tensor(pred_idx, dtype=torch.long),
            torch.as_tensor(gt_idx,   dtype=torch.long),
        )


def match_and_compute_loss(
    pred_boxes_ri:    Tensor,    # (B, Q, 4)
    pred_logits:      Tensor,    # (B, Q, C)
    gt_boxes_list:    list,      # list of B tensors (M_b, 4) — variable M per image
    gt_labels_list:   list,      # list of B tensors (M_b,)
    o:                Tensor,    # (B, 2)
    R:                Tensor,    # (B,)
    matcher:          HungarianMatcher,
    num_classes:      int,
    alpha:            float = 2.0,
    p:                float = 2.0,
    lambda_l1:        float = 5.0,
    lambda_giou:      float = 2.0,
    lambda_cls:       float = 1.0,
    eos_coef:         float = 0.1,
) -> dict:
    """Full loss computation with Hungarian matching for a batch.

    This replaces the random-assignment placeholder in Notebook 05.

    Args:
        pred_boxes_ri   : (B, Q, 4) Riemannian predictions
        pred_logits     : (B, Q, C) class logits
        gt_boxes_list   : list of (M_b, 4) GT Euclidean boxes per image
        gt_labels_list  : list of (M_b,)   GT class labels per image
        o, R            : foveation parameters per image
    Returns:
        dict with loss_total, loss_l1, loss_giou, loss_cls
    """
    from .box import riemannian_to_euclidean_box
    from .head import generalized_box_iou

    B, Q, _ = pred_boxes_ri.shape
    bg_class = num_classes - 1   # background = last class index

    # Down-weight the no-object class (DETR eos_coef): most of the Q queries match
    # background, so without this the CE collapses to always predicting "no object".
    cls_weight = pred_logits.new_ones(num_classes)
    cls_weight[bg_class] = eos_coef

    total_l1   = pred_boxes_ri.new_zeros(1)
    total_giou = pred_boxes_ri.new_zeros(1)
    total_cls  = pred_boxes_ri.new_zeros(1)
    n_matched  = 0

    for b in range(B):
        gt_b   = gt_boxes_list[b].to(pred_boxes_ri.device)    # (M, 4)
        lbl_b  = gt_labels_list[b].to(pred_logits.device)     # (M,)
        M      = gt_b.shape[0]

        # ── Hungarian matching ────────────────────────────────────────────────
        pred_idx, gt_idx = matcher(
            pred_boxes_ri[b], pred_logits[b],
            gt_b, lbl_b, o[b], R[b], alpha, p
        )

        # ── Classification loss (all Q queries) ───────────────────────────────
        # Unmatched queries → background class
        target_labels = torch.full((Q,), bg_class, dtype=torch.long,
                                    device=pred_logits.device)
        target_labels[pred_idx] = lbl_b[gt_idx]
        total_cls = total_cls + F.cross_entropy(pred_logits[b], target_labels,
                                                weight=cls_weight)

        if len(pred_idx) == 0:
            continue

        # ── Box losses (matched pairs only) ──────────────────────────────────
        pred_matched_ri = pred_boxes_ri[b, pred_idx]    # (K, 4)
        gt_matched_eu   = gt_b[gt_idx]                  # (K, 4)

        # Decode to Euclidean for L1 and gIoU
        pred_matched_eu = riemannian_to_euclidean_box(
            pred_matched_ri, o[b], R[b], alpha, p
        )

        total_l1   = total_l1   + F.l1_loss(pred_matched_eu, gt_matched_eu)
        giou_vals  = generalized_box_iou(pred_matched_eu, gt_matched_eu)
        total_giou = total_giou + (1 - giou_vals).mean()
        n_matched  += len(pred_idx)

    # Average over batch
    loss_l1   = total_l1   / B
    loss_giou = total_giou / B
    loss_cls  = total_cls  / B
    loss_total = lambda_l1 * loss_l1 + lambda_giou * loss_giou + lambda_cls * loss_cls

    return dict(
        loss_total=loss_total,
        loss_l1=loss_l1,
        loss_giou=loss_giou,
        loss_cls=loss_cls,
        n_matched=n_matched,
    )


def compute_denoising_loss(
    dn_out:           dict,        # from TelescopeModel._denoising_pass
    gt_boxes_list:    list,        # list of (M_b, 4) Euclidean GT boxes
    gt_labels_list:   list,        # list of (M_b,)   GT labels
    o:                Tensor,      # (B, 2)
    R:                Tensor,      # (B,)
    num_classes:      int,
    alpha:            float = 2.0,
    p:                float = 2.0,
    lambda_l1:        float = 5.0,
    lambda_giou:      float = 2.0,
    lambda_cls:       float = 1.0,
) -> dict:
    """DINO-style denoising loss (no Hungarian matching).

    Each denoising query corresponds 1-to-1 to a GT box, so we supervise it
    directly: decode the predicted Riemannian box to Euclidean and apply the
    same L1 + gIoU + classification losses against its GT.  This gives a dense,
    stable training signal that accelerates convergence (Telescope paper §4).

    Args:
        dn_out : dict with 'boxes_ri' (B,M,4), 'logits' (B,M,C), 'mask' (B,M)
    Returns:
        dict with loss_dn, loss_dn_l1, loss_dn_giou, loss_dn_cls, n_dn
    """
    from .box import riemannian_to_euclidean_box
    from .head import generalized_box_iou

    boxes_ri = dn_out["boxes_ri"]    # (B, M, 4)
    logits   = dn_out["logits"]      # (B, M, C)
    mask     = dn_out["mask"]        # (B, M)
    B = boxes_ri.shape[0]

    total_l1   = boxes_ri.new_zeros(1)
    total_giou = boxes_ri.new_zeros(1)
    total_cls  = boxes_ri.new_zeros(1)
    n_dn = 0

    for b in range(B):
        m = int(mask[b].sum())
        if m == 0:
            continue
        gt  = gt_boxes_list[b].to(boxes_ri.device)[:m]    # (m, 4)
        lbl = gt_labels_list[b].to(logits.device)[:m]     # (m,)

        pred_eu = riemannian_to_euclidean_box(boxes_ri[b, :m], o[b], R[b], alpha, p)
        total_l1   = total_l1   + F.l1_loss(pred_eu, gt)
        total_giou = total_giou + (1.0 - generalized_box_iou(pred_eu, gt)).mean()
        total_cls  = total_cls  + F.cross_entropy(logits[b, :m], lbl)
        n_dn += m

    denom    = max(B, 1)
    loss_l1  = total_l1   / denom
    loss_giou= total_giou / denom
    loss_cls = total_cls  / denom
    loss_dn  = lambda_l1 * loss_l1 + lambda_giou * loss_giou + lambda_cls * loss_cls

    return dict(
        loss_dn=loss_dn,
        loss_dn_l1=loss_l1,
        loss_dn_giou=loss_giou,
        loss_dn_cls=loss_cls,
        n_dn=n_dn,
    )
