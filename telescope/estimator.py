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

    def __init__(self, in_features: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),   # → [o_x_logit, o_y_logit, R_x_logit, R_y_logit]
        )

    def forward(self, features: Tensor):
        """Predict foveation parameters from encoder features.

        Args:
            features : (B, in_features)
        Returns:
            o : (B, 2) foveation centre in (−1, 1)²
            R : (B,)   radial scale > 0
        """
        logits = self.mlp(features)                           # (B, 4)
        o      = torch.tanh(logits[:, :2])                    # (B, 2)
        R_xy   = F.softplus(logits[:, 2:])                    # (B, 2) > 0
        R      = R_xy.max(dim=-1).values                      # (B,)
        return o, R
