"""
telescope.box
=============
Riemannian bounding-box encode/decode.   Paper §3.2.

A Euclidean box  b  = [cx, cy, w, h]
maps to a Riemannian box  b' = [Φ(c), ‖t_x‖, ‖t_y‖]
where  t_x = J_Φ(c)·[w, 0]ᵀ  and  t_y = J_Φ(c)·[0, h]ᵀ.

This parameterisation lets the detector predict boxes directly in the
induced Riemannian space, avoiding axis-aligned assumptions that break
under non-linear warps.
"""

import torch
from torch import Tensor
from .geometry import EPS, hyperbolic_foveated_transform, compute_jacobian, hyperbolic_inverse

__all__ = ["euclidean_to_riemannian_box", "riemannian_to_euclidean_box"]


def euclidean_to_riemannian_box(
    boxes: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
) -> Tensor:
    """Encode Euclidean boxes → Riemannian boxes.

    b = [cx, cy, w, h]  →  b' = [Φ_x(c), Φ_y(c), ‖t_x‖, ‖t_y‖]

    where
        t_x = J_Φ(c) · [w, 0]ᵀ   (how a horizontal step of size w warps)
        t_y = J_Φ(c) · [0, h]ᵀ   (how a vertical  step of size h warps)

    Args:
        boxes : (N, 4)  [cx, cy, w, h] in normalised [-1, 1]² coords
        o, R  : transform parameters
    Returns:
        (N, 4) Riemannian boxes
    """
    c = boxes[:, :2]           # (N, 2) box centres
    w = boxes[:, 2:3]          # (N, 1) widths
    h = boxes[:, 3:4]          # (N, 1) heights

    phi_c = hyperbolic_foveated_transform(c, o, R, alpha, p)   # (N, 2)
    J     = compute_jacobian(c, o, R, alpha, p)                # (N, 2, 2)

    # cat along dim=-1: (N,1)||(N,1) → (N,2)  [stack would give (N,1,2)]
    e_x = torch.cat([w, torch.zeros_like(w)], dim=-1)          # (N, 2)
    e_y = torch.cat([torch.zeros_like(h), h], dim=-1)          # (N, 2)

    # (N,2,2) @ (N,2,1) → (N,2,1) → (N,2)
    t_x = (J @ e_x.unsqueeze(-1)).squeeze(-1)
    t_y = (J @ e_y.unsqueeze(-1)).squeeze(-1)

    tx_norm = torch.norm(t_x, dim=-1, keepdim=True)            # (N, 1)
    ty_norm = torch.norm(t_y, dim=-1, keepdim=True)            # (N, 1)

    return torch.cat([phi_c, tx_norm, ty_norm], dim=-1)        # (N, 4)


def riemannian_to_euclidean_box(
    boxes_ri: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
) -> Tensor:
    """Decode Riemannian boxes → Euclidean boxes.   Inverse of above.

    Used at loss time: predicted b' is projected back to Euclidean space
    so standard L1 / gIoU losses can be computed.

    Recovery of w, h:
        t_x = J_Φ(c) · [w, 0]ᵀ = J_Φ(c)[:,0] · w
        ‖t_x‖ = ‖J_Φ(c)[:,0]‖ · w   (w > 0)
        ⇒  w = ‖t_x‖ / ‖J_Φ(c)[:,0]‖

    Args:
        boxes_ri : (N, 4)  [Φ_x(c), Φ_y(c), ‖t_x‖, ‖t_y‖]
    Returns:
        (N, 4) Euclidean boxes  [cx, cy, w, h]
    """
    phi_c   = boxes_ri[:, :2]    # (N, 2)
    tx_norm = boxes_ri[:, 2:3]   # (N, 1)
    ty_norm = boxes_ri[:, 3:4]   # (N, 1)

    # Recover centre
    c = hyperbolic_inverse(phi_c, o, R, alpha, p)              # (N, 2)

    # Recover w, h from column norms of J_Φ(c)
    J         = compute_jacobian(c, o, R, alpha, p)            # (N, 2, 2)
    col0_norm = torch.norm(J[:, :, 0], dim=-1, keepdim=True)   # (N, 1)
    col1_norm = torch.norm(J[:, :, 1], dim=-1, keepdim=True)   # (N, 1)

    w = tx_norm / col0_norm.clamp(min=EPS)
    h = ty_norm / col1_norm.clamp(min=EPS)

    return torch.cat([c, w, h], dim=-1)                        # (N, 4)
