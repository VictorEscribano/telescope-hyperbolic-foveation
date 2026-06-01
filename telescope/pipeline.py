"""
telescope.pipeline
==================
TelescopeModel: full two-stage detection pipeline.

Stage 1  — Hyperbolic foveation (this codebase):
    - FoveationEstimator  → (o, R)
    - FoveationWarpLayer  → warped image
    - HyperbolicEmbedding → query context

Stage 2a — Detection (requires external backbone + DETR):
    - SAM3 image encoder  (frozen, from HuggingFace / Torc Robotics)
    - Deformable DETR encoder + decoder
    - RiemannianBoxHead   → predicted b' in Riemannian space

Stage 2b — Re-projection:
    - Phi^{-1} (NR inverse) → Euclidean bounding boxes

In this implementation SAM3 and Deformable DETR are represented by
lightweight stubs so the full pipeline can be tested without GPU-heavy
weights.  See README.md for instructions on substituting real weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .estimator import FoveationEstimator
from .warp import FoveationWarpLayer
from .embedding import HyperbolicEmbedding, augment_queries
from .head import RiemannianBoxHead, TelescopeLoss, denoise_boxes
from .box import riemannian_to_euclidean_box

__all__ = [
    "SAM3EncoderStub",
    "DeformableDetrStub",
    "TelescopeModel",
]


# ── SAM3 encoder stub ─────────────────────────────────────────────────────────

class SAM3EncoderStub(nn.Module):
    """Lightweight stand-in for the frozen SAM3 image encoder.

    Real SAM3 (Segment Anything Model 3):
        - ViT backbone with 32 layers, embedding dim 1024, patch size 14
        - Windowed local self-attention every 7th layer + global attention
        - SimpleFPN neck → 4× / 2× / 1× feature maps of channel dim 256

    This stub reproduces the output SHAPES so the rest of the pipeline
    can be developed and tested without downloading SAM3 weights.

    Real weights: see README.md → 'Downloading pre-trained models'.
    """

    def __init__(self, out_channels: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        # 1x1 conv maps 3→out_channels while preserving spatial dims (FPN features).
        # Explicit in_channels avoids the nn.LazyConv2d uninitialized-parameter error.
        self.fpn_4x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fpn_2x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fpn_1x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, image: Tensor):
        """
        Args:
            image : (B, 3, H, W)
        Returns:
            features : list of 3 tensors at 4×, 2×, 1× resolution reductions
                       [(B, C, H/4, W/4), (B, C, H/2, W/2), (B, C, H, W)]
        """
        H, W = image.shape[-2:]
        f4 = F.interpolate(image, (H // 4, W // 4), mode='bilinear', align_corners=True)
        f2 = F.interpolate(image, (H // 2, W // 2), mode='bilinear', align_corners=True)
        f1 = image
        return [self.fpn_4x(f4), self.fpn_2x(f2), self.fpn_1x(f1)]


# ── Deformable DETR stub ──────────────────────────────────────────────────────

class DeformableDetrStub(nn.Module):
    """Lightweight stand-in for the Deformable DETR encoder + decoder.

    Real Deformable DETR (Zhu et al., 2020):
        - Multi-scale deformable attention encoder (6 layers)
        - Deformable attention decoder (6 layers, 4 sampling points / level)
        - Queries: 256-d learned embeddings, 300 queries

    This stub uses standard multi-head attention as an approximation — enough
    to verify shapes and gradients without the deformable attention kernels.

    To replace with the real implementation (see README.md):
        pip install transformers

        from transformers import DeformableDetrConfig, DeformableDetrModel
        config = DeformableDetrConfig(d_model=256, encoder_layers=6,
                                       decoder_layers=6, num_queries=300,
                                       num_feature_levels=3, decoder_n_points=4)
        detr = DeformableDetrModel(config)
    """

    def __init__(
        self,
        query_dim:   int = 256,
        num_queries: int = 300,
        num_heads:   int = 8,
        num_layers:  int = 2,   # real DETR uses 6
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_dim   = query_dim

        # Project multi-scale features to query_dim
        # Input dim = query_dim because FPN outputs query_dim-channel tokens
        self.feat_proj = nn.Linear(query_dim, query_dim)

        # Simplified decoder: cross-attention between queries and flattened features
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=query_dim, nhead=num_heads, dim_feedforward=query_dim * 4,
                batch_first=True, dropout=0.0,
            ),
            num_layers=num_layers,
        )

    def forward(self, features: list, queries: Tensor) -> Tensor:
        """
        Args:
            features : list of FPN feature tensors [(B, C, H_i, W_i)]
            queries  : (B, num_queries, query_dim) — augmented object queries
        Returns:
            (B, num_queries, query_dim) decoded query features
        """
        # Flatten and concatenate all scale features → (B, total_tokens, C)
        tokens = []
        for f in features:
            B, C, H, W = f.shape
            tokens.append(f.flatten(2).permute(0, 2, 1))   # (B, H*W, C)
        memory = torch.cat(tokens, dim=1)                   # (B, total_tokens, C)
        memory = self.feat_proj(memory)                     # (B, total_tokens, query_dim)

        return self.decoder(queries, memory)                # (B, num_queries, query_dim)


# ── Full Telescope model ──────────────────────────────────────────────────────

class TelescopeModel(nn.Module):
    """Complete two-stage Telescope detection model.

    Forward pass:
        image → [Stage 1] warp + embedding → [Stage 2a] encode + decode
             → [RiemannianBoxHead] b' → [Stage 2b] Phi^{-1} → Euclidean boxes

    Args:
        num_classes  : number of object categories + 1 (background)
        num_queries  : number of DETR object queries  [paper: 300]
        query_dim    : query / embedding dimension    [paper: 256]
        enc_out_dim  : FoveationEstimator feature dim [paper: 256]
        image_size   : (H, W) for the high-res inference image [paper: 1024×1024]
        low_res_size : (H, W) for the param-estimation image   [paper: 512×512]
    """

    def __init__(
        self,
        num_classes:  int = 6,
        num_queries:  int = 300,
        query_dim:    int = 256,
        enc_out_dim:  int = 256,
    ) -> None:
        super().__init__()

        # ── Stage 1 ──────────────────────────────────────────────────────────
        self.param_encoder   = SAM3EncoderStub(out_channels=enc_out_dim)
        self.fov_estimator   = FoveationEstimator(enc_out_dim * 3, hidden=enc_out_dim)
        self.warp_layer      = FoveationWarpLayer(alpha=2.0, p=2.0)
        self.hyperbolic_emb  = HyperbolicEmbedding(param_dim=4, query_dim=query_dim)

        # ── Stage 2a ─────────────────────────────────────────────────────────
        self.backbone        = SAM3EncoderStub(out_channels=query_dim)
        self.detr            = DeformableDetrStub(query_dim, num_queries)
        self.object_queries  = nn.Embedding(num_queries, query_dim)

        # ── Stage 2b ─────────────────────────────────────────────────────────
        self.box_head        = RiemannianBoxHead(query_dim, num_classes)

        self.alpha = 2.0
        self.p     = 2.0

    def forward(self, image: Tensor, return_riemannian: bool = False):
        """Full forward pass.

        Args:
            image             : (B, 3, H, W)
            return_riemannian : if True, also return Riemannian boxes b'
        Returns:
            boxes_eu    : (B, num_queries, 4) Euclidean [cx, cy, w, h]
            class_logits: (B, num_queries, num_classes)
            o, R        : foveation parameters (for loss / logging)
            boxes_ri    : (B, num_queries, 4) Riemannian b'  [only if return_riemannian]
        """
        B = image.shape[0]

        # ── Stage 1a: estimate foveation params from low-res image ────────────
        small      = F.interpolate(image, (64, 64), mode='bilinear', align_corners=True)
        feats_low  = self.param_encoder(small)             # 3 FPN tensors
        # Global pool each scale and concatenate
        enc_vec    = torch.cat([f.flatten(2).mean(-1) for f in feats_low], dim=-1)  # (B, C*3)
        o, R       = self.fov_estimator(enc_vec)           # (B,2), (B,)

        # ── Stage 1b: warp full-resolution image ──────────────────────────────
        warped     = self.warp_layer(image, o, R)          # (B, 3, H, W)

        # ── Stage 1c: hyperbolic embedding → augment queries ─────────────────
        fov_params  = torch.cat([o, R.unsqueeze(-1).expand(-1, 2)], dim=-1)  # (B,4)
        embedding   = self.hyperbolic_emb(fov_params)      # (B, query_dim)
        queries     = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)
        aug_queries = augment_queries(queries, embedding)  # (B, Q, D)

        # ── Stage 2a: encode warped image + decode with augmented queries ─────
        features    = self.backbone(warped)                # 3-scale FPN features
        query_feats = self.detr(features, aug_queries)     # (B, Q, D)

        # ── Stage 2b: predict + re-project ───────────────────────────────────
        boxes_ri, class_logits = self.box_head(query_feats)  # (B,Q,4), (B,Q,C)

        # Inverse-project each image's boxes using its own (o_b, R_b)
        boxes_eu_list = []
        for b in range(B):
            eu = riemannian_to_euclidean_box(
                boxes_ri[b], o[b], R[b], self.alpha, self.p
            )
            boxes_eu_list.append(eu)
        boxes_eu = torch.stack(boxes_eu_list, dim=0)       # (B, Q, 4)

        if return_riemannian:
            return boxes_eu, class_logits, o, R, boxes_ri
        return boxes_eu, class_logits, o, R
