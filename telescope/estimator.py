"""
telescope.estimator
===================
FoveationEstimator: small FFN that predicts (o, R) from encoder features.

In the full Telescope pipeline (paper §4), this is driven by the output
of the SAM3 image encoder on a 256×256 or 512×512 downsampled image.
Here it is implemented as a standalone module so it can be tested and
replaced independently of the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["FoveationEstimator"]


class FoveationEstimator(nn.Module):
    """3-layer MLP that predicts foveation parameters (o, R) from a feature vector.

    Architecture (paper §4 / Appendix D):
        Linear(in_features, hidden) → ReLU
        Linear(hidden, hidden)      → ReLU
        Linear(hidden, 4)           → [o_x_logit, o_y_logit, R_x_logit, R_y_logit]

    Output activations:
        o = tanh(logit)          — keeps centre inside (−1, 1)²
        R = softplus(logit)      — keeps radius strictly positive with smooth grad
        R = max(R_x, R_y)        — scalar radius as per Appendix D

    To use with a real SAM3 backbone, pass the 1× feature map flattened or
    globally pooled as `features`.  Replace `in_features` with the actual dim.

    Args:
        in_features : dimension of the input feature vector
        hidden      : width of the two hidden layers  [paper: 256]
    """

    def __init__(self, in_features: int = 256, hidden: int = 256,
                 feat_ch: int = 256, spatial_o: bool = False) -> None:
        super().__init__()
        self.spatial_o = spatial_o
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),   # → [o_x_logit, o_y_logit, R_x_logit, R_y_logit]
        )
        # Spatial head for `o`: a 1×1 conv produces a per-location score map; a
        # soft-argmax over it gives a centre that CAN vary per image.  Without
        # this, `o` comes from globally-pooled features (no spatial info) and can
        # only learn a constant — the collapse observed in drones_et65/et66.
        if spatial_o:
            self.o_head = nn.Sequential(
                nn.Conv2d(feat_ch, hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )

    def _soft_argmax(self, feat_map: Tensor) -> Tensor:
        """Soft-argmax of a learned heatmap → centre o in (−1, 1)².

        Args:
            feat_map : (B, C, H, W) finest encoder feature map
        Returns:
            (B, 2) in (−1, 1)²  (computed in fp32 for a stable softmax)
        """
        heat = self.o_head(feat_map).float()                  # (B, 1, H, W)
        B, _, H, W = heat.shape
        prob = F.softmax(heat.view(B, -1), dim=-1).view(B, 1, H, W)
        xs = torch.linspace(-1.0, 1.0, W, device=heat.device, dtype=heat.dtype)
        ys = torch.linspace(-1.0, 1.0, H, device=heat.device, dtype=heat.dtype)
        px = prob.sum(dim=2).squeeze(1)                       # (B, W) marginal over rows
        py = prob.sum(dim=3).squeeze(1)                       # (B, H) marginal over cols
        o_x = (px * xs).sum(dim=-1)
        o_y = (py * ys).sum(dim=-1)
        return torch.stack([o_x, o_y], dim=-1)                # (B, 2)

    def forward(self, features: Tensor, feat_map: Tensor = None):
        """Predict foveation parameters from encoder features.

        Args:
            features : (B, in_features)  globally-pooled feature vector (drives R)
            feat_map : (B, C, H, W)      finest feature map (drives spatial o);
                                         required when ``spatial_o=True``
        Returns:
            o : (B, 2) foveation centre in (−1, 1)²
            R : (B,)   radial scale > 0
        """
        logits = self.mlp(features)                           # (B, 4)
        R_xy   = F.softplus(logits[:, 2:])                    # (B, 2) > 0
        R      = R_xy.max(dim=-1).values                      # (B,)
        if self.spatial_o and feat_map is not None:
            o = self._soft_argmax(feat_map).to(logits.dtype)  # (B, 2) per-image
        else:
            o = torch.tanh(logits[:, :2])                     # (B, 2)
        return o, R
