"""
telescope.embedding
===================
HyperbolicEmbedding: projects foveation parameters into DETR query space.

Why this exists
---------------
The Deformable DETR head predicts bounding boxes in the *warped* (Riemannian)
image space.  Without knowing the current warp, a query at position [0.1, 0.2]
cannot tell whether it is near the foveation centre (heavily magnified) or
far from it (near identity).

HyperbolicEmbedding solves this by projecting the 4-dimensional parameter
vector [o_x, o_y, R_x, R_y] into a query_dim-dimensional vector that gets
*added* to every DETR object query before the decoder.  This is conceptually
identical to positional encoding in ViT: it injects geometric context that
the model cannot derive from the image features alone.

Reference: Telescope paper §4 / Appendix D Table 9 "Foveation Embed" row.
"""

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["HyperbolicEmbedding"]


class HyperbolicEmbedding(nn.Module):
    """3-layer MLP that maps foveation parameters → DETR query embedding.

    Architecture (paper Appendix D, Table 9 "Foveation Embed"):
        Linear(param_dim → hidden) → ReLU
        Linear(hidden → hidden)    → ReLU
        Linear(hidden → query_dim)

    The output is added (broadcast) to all object queries:
        queries  : (B, num_queries, query_dim)
        embedding: (B, query_dim) → unsqueeze → (B, 1, query_dim)
        result   : (B, num_queries, query_dim)   ← same shape

    Args:
        param_dim : dimension of the input parameter vector.
                    Default 4 = [o_x, o_y, R_x, R_y].
                    Can also include alpha and p (→ 6) if those are learned.
        query_dim : DETR query / embedding dimension.
                    Default 256 matches Deformable DETR.
        hidden    : width of the two hidden layers.  Default 256.
    """

    def __init__(
        self,
        param_dim: int = 4,
        query_dim: int = 256,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(param_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, query_dim),
        )

    def forward(self, params: Tensor) -> Tensor:
        """Project foveation parameters to query embedding space.

        Args:
            params : (B, param_dim)  —  e.g. [o_x, o_y, R_x, R_y]
        Returns:
            (B, query_dim) embedding, ready to be broadcast-added to queries
        """
        return self.mlp(params)   # (B, query_dim)


def augment_queries(
    queries: Tensor,
    embedding: Tensor,
) -> Tensor:
    """Add a foveation embedding to all object queries.

    This is how the detector learns about the current warp:
    the same geometric context is injected into every query.

    Args:
        queries   : (B, num_queries, query_dim)
        embedding : (B, query_dim)
    Returns:
        (B, num_queries, query_dim)
    """
    return queries + embedding.unsqueeze(1)   # broadcast over num_queries
