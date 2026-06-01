"""
telescope
=========
Learnable Hyperbolic Foveation for Ultra-Long-Range Object Detection.
Re-implementation of Ewen et al. (2026), arXiv:2604.06332.

Submodules
----------
geometry   — Φ, Φ⁻¹, Jacobian (the mathematical core)
box        — Riemannian bounding-box encode/decode
warp       — differentiable image warp layer
estimator  — FoveationEstimator FFN (predicts o, R from encoder features)
"""

from .geometry import (
    poincare_projection,
    blend_weight,
    hyperbolic_foveated_transform,
    compute_jacobian,
    HyperbolicInverseNR,
    hyperbolic_inverse,
    validate_inversion,
)
from .box import euclidean_to_riemannian_box, riemannian_to_euclidean_box
from .warp import create_output_grid, compute_source_grid, FoveationWarpLayer
from .estimator import FoveationEstimator
from .embedding import HyperbolicEmbedding, augment_queries
from .head import RiemannianBoxHead, TelescopeLoss, denoise_boxes, generalized_box_iou
from .pipeline import SAM3EncoderStub, DeformableDetrStub, TelescopeModel
from .matcher import HungarianMatcher, match_and_compute_loss
from .eval import CocoEvaluator, DetectionResult
from .data import Argoverse2Dataset, collate_fn
from .checkpoint import CheckpointManager

__all__ = [
    "poincare_projection",
    "blend_weight",
    "hyperbolic_foveated_transform",
    "compute_jacobian",
    "HyperbolicInverseNR",
    "hyperbolic_inverse",
    "validate_inversion",
    "euclidean_to_riemannian_box",
    "riemannian_to_euclidean_box",
    "create_output_grid",
    "compute_source_grid",
    "FoveationWarpLayer",
    "FoveationEstimator",
    "HyperbolicEmbedding",
    "augment_queries",
    "RiemannianBoxHead",
    "TelescopeLoss",
    "denoise_boxes",
    "generalized_box_iou",
    "SAM3EncoderStub",
    "DeformableDetrStub",
    "TelescopeModel",
    "HungarianMatcher",
    "match_and_compute_loss",
    "CocoEvaluator",
    "DetectionResult",
]
