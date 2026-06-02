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
from .head import RiemannianBoxHead
from .box import riemannian_to_euclidean_box

__all__ = [
    "SAM3EncoderStub",
    "DeformableDetrStub",
    "RealDeformableDetr",
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

    def forward(self, features: list, queries: Tensor,
                reference_points: Tensor = None) -> Tensor:
        """
        Args:
            features : list of FPN feature tensors [(B, C, H_i, W_i)]
                       Only the two coarsest scales (4× and 2× downsampled) are
                       used — matching the real Deformable DETR which never
                       attends to full-resolution features.  Using the 1× level
                       creates O(H×W) tokens and blows up cross-attention memory
                       on images ≥ 640px.
            queries  : (B, num_queries, query_dim) — augmented object queries
            reference_points : unused by the stub (accepted for API parity with
                       RealDeformableDetr's denoising path).
        Returns:
            (B, num_queries, query_dim) decoded query features
        """
        return self.decode(self.encode(features), queries, reference_points)

    def encode(self, features: list) -> dict:
        """Flatten the two coarsest FPN levels into a memory bank (run once)."""
        tokens = []
        for f in features[:2]:
            B, C, H, W = f.shape
            tokens.append(f.flatten(2).permute(0, 2, 1))   # (B, H*W, C)
        memory = torch.cat(tokens, dim=1)                   # (B, total_tokens, C)
        memory = self.feat_proj(memory)                     # (B, total_tokens, query_dim)
        return {"memory": memory}

    def decode(self, ctx: dict, queries: Tensor,
               reference_points: Tensor = None) -> Tensor:
        """Cross-attend queries to the pre-computed memory (reference_points
        unused by the stub; accepted for API parity)."""
        return self.decoder(queries, ctx["memory"])         # (B, num_queries, query_dim)


# ── Real Deformable DETR (transformers ≥ 4.40) ────────────────────────────────

class RealDeformableDetr(nn.Module):
    """
    Drop-in replacement for DeformableDetrStub using the real encoder + decoder
    from the ``transformers`` library.

    Key differences from the stub:
    - Multi-scale deformable self-attention in the encoder (O(N) vs O(N²)).
    - Deformable cross-attention in the decoder — samples only 4 points per
      query per FPN level instead of attending to all tokens.
    - 6 encoder + 6 decoder layers matching the paper (stub uses 2).
    - Per-level positional embedding distinguishes the two FPN scales.

    Forward signature is identical to DeformableDetrStub so TelescopeModel
    uses this transparently.
    """

    def __init__(
        self,
        query_dim:          int = 256,
        num_queries:        int = 300,
        num_feature_levels: int = 2,
    ) -> None:
        super().__init__()
        from transformers import DeformableDetrConfig
        from transformers.models.deformable_detr.modeling_deformable_detr import (
            DeformableDetrEncoder,
            DeformableDetrDecoder,
            DeformableDetrSinePositionEmbedding,
        )

        self.num_feature_levels = num_feature_levels
        self.query_dim          = query_dim

        cfg = DeformableDetrConfig(
            d_model             = query_dim,
            encoder_layers      = 6,
            decoder_layers      = 6,
            num_queries         = num_queries,
            num_feature_levels  = num_feature_levels,
            encoder_n_points    = 4,
            decoder_n_points    = 4,
            two_stage           = False,
        )

        # One conv+norm projection per FPN level to normalise features before
        # the encoder.  FPN already outputs query_dim channels, but the GroupNorm
        # stabilises training.
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(query_dim, query_dim, kernel_size=1),
                nn.GroupNorm(32, query_dim),
            )
            for _ in range(num_feature_levels)
        ])

        # Per-level embedding lets the encoder distinguish FPN scales (mirrors
        # DeformableDetrModel.level_embed).
        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, query_dim))
        nn.init.normal_(self.level_embed)

        self.pos_emb  = DeformableDetrSinePositionEmbedding(query_dim // 2, normalize=True)
        self.encoder  = DeformableDetrEncoder(cfg)
        self.decoder  = DeformableDetrDecoder(cfg)

        # Projects augmented query content to (cx, cy) reference points in [0,1].
        # The decoder internally expands these to all FPN levels via valid_ratios.
        self.ref_pts  = nn.Linear(query_dim, 2)

    def encode(self, features: list) -> dict:
        """Run the deformable encoder once and return a reusable context.

        Splitting encode/decode lets the DINO-style denoising pass reuse the
        (expensive) encoder output instead of recomputing it.
        """
        device = features[0].device

        src_flat_list  = []
        pos_flat_list  = []
        spatial_shapes_list = []

        for i, feat in enumerate(features[:self.num_feature_levels]):
            src = self.input_proj[i](feat)          # (B, query_dim, H_i, W_i)
            b, c, h, w = src.shape
            spatial_shapes_list.append((h, w))

            # Sine positional encoding.  transformers<5 returns (B, C, H, W);
            # transformers>=5 returns it already flattened to (B, H*W, C).
            pos = self.pos_emb(shape=src.shape, device=device, dtype=src.dtype)
            if pos.dim() == 4:                          # (B, C, H, W) → (B, H*W, C)
                pos = pos.flatten(2).permute(0, 2, 1)

            # Flatten spatial dims and add level embedding
            src_flat = src.flatten(2).permute(0, 2, 1)              # (B, H*W, C)
            pos_flat = pos + self.level_embed[i]                    # (B, H*W, C)

            src_flat_list.append(src_flat)
            pos_flat_list.append(pos_flat)

        src_flat = torch.cat(src_flat_list, dim=1)  # (B, total_tokens, C)
        pos_flat = torch.cat(pos_flat_list, dim=1)  # (B, total_tokens, C)

        spatial_shapes  = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=device)
        level_start_idx = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ])
        # All pixels are valid (no padding in our batches).
        valid_ratios = torch.ones(src_flat.shape[0], self.num_feature_levels, 2, device=device)

        enc_out = self.encoder(
            inputs_embeds              = src_flat,
            spatial_position_embeddings = pos_flat,
            spatial_shapes             = spatial_shapes,
            spatial_shapes_list        = spatial_shapes_list,
            level_start_index          = level_start_idx,
            valid_ratios               = valid_ratios,
        ).last_hidden_state                          # (B, total_tokens, C)

        return {
            "enc_out":             enc_out,
            "spatial_shapes":      spatial_shapes,
            "spatial_shapes_list": spatial_shapes_list,
            "level_start_index":   level_start_idx,
            "valid_ratios":        valid_ratios,
        }

    def decode(self, ctx: dict, queries: Tensor,
               reference_points: Tensor = None) -> Tensor:
        """Run the deformable decoder against a pre-computed encoder context.

        Args:
            ctx              : output of :meth:`encode`
            queries          : (B, Q, query_dim) object/denoising queries
            reference_points : (B, Q, 2) in [0,1].  If given (denoising), used
                               directly; else predicted from query content.
        """
        # Reference points (cx, cy) ∈ [0,1].  Normally derived from query content;
        # for DINO-style denoising the caller supplies the noised GT centres.
        if reference_points is None:
            ref_pts = self.ref_pts(queries).sigmoid()   # (B, Q, 2)
        else:
            ref_pts = reference_points

        return self.decoder(
            inputs_embeds                       = queries,
            object_queries_position_embeddings  = None,   # foveation emb. already in queries
            encoder_hidden_states               = ctx["enc_out"],
            reference_points                    = ref_pts,
            spatial_shapes                      = ctx["spatial_shapes"],
            spatial_shapes_list                 = ctx["spatial_shapes_list"],
            level_start_index                   = ctx["level_start_index"],
            valid_ratios                        = ctx["valid_ratios"],
        ).last_hidden_state                          # (B, Q, C)

    def forward(self, features: list, queries: Tensor,
                reference_points: Tensor = None) -> Tensor:
        """Encode features then decode queries (single-pass convenience)."""
        return self.decode(self.encode(features), queries, reference_points)


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
        low_res_size: int = 512,
    ) -> None:
        super().__init__()

        # ── Stage 1 ──────────────────────────────────────────────────────────
        # Foveation params (o, R) are estimated from the *shared* detection
        # backbone run on a low-res, unwarped image (paper §4: "a small FFN
        # processes encoder output from low-resolution images (256×256 or
        # 512×512)").  No separate param encoder — sharing keeps the estimate on
        # real SAM3 features instead of a random stub, and uses the paper's 512²
        # rather than a 64² thumbnail.
        self.low_res_size    = low_res_size
        self.fov_estimator   = FoveationEstimator(query_dim * 3, hidden=enc_out_dim)
        self.warp_layer      = FoveationWarpLayer(alpha=2.0, p=2.0)
        self.hyperbolic_emb  = HyperbolicEmbedding(param_dim=4, query_dim=query_dim)

        # ── Stage 2a ─────────────────────────────────────────────────────────
        self.backbone = SAM3EncoderStub(out_channels=query_dim)
        try:
            self.detr = RealDeformableDetr(query_dim, num_queries)
            print("[telescope] Using real Deformable DETR (transformers)")
        except ImportError:
            self.detr = DeformableDetrStub(query_dim, num_queries)
            print("[telescope] transformers not found — using DeformableDetrStub")
        self.object_queries = nn.Embedding(num_queries, query_dim)

        # DINO-style denoising: per-class content embedding for noised GT queries.
        self.dn_label_emb = nn.Embedding(num_classes, query_dim)

        # ── Stage 2b ─────────────────────────────────────────────────────────
        self.box_head        = RiemannianBoxHead(query_dim, num_classes)

        self.num_classes = num_classes
        self.query_dim   = query_dim
        self.alpha = 2.0
        self.p     = 2.0

    def forward(self, image: Tensor, return_riemannian: bool = False,
                denoising: tuple = None):
        """Full forward pass.

        Args:
            image             : (B, 3, H, W)
            return_riemannian : if True, also return Riemannian boxes b'
            denoising         : optional ``(gt_boxes_list, gt_labels_list,
                                noise_scale)``.  When given (training only), a
                                second decoder pass refines noised GT boxes
                                (DINO-style) and an extra ``dn_out`` dict is
                                appended to the return tuple.
        Returns:
            boxes_eu    : (B, num_queries, 4) Euclidean [cx, cy, w, h]
            class_logits: (B, num_queries, num_classes)
            o, R        : foveation parameters (for loss / logging)
            boxes_ri    : (B, num_queries, 4) Riemannian b'  [only if return_riemannian]
            dn_out      : dict with denoising preds      [only if denoising given]
        """
        B = image.shape[0]

        # ── Stage 1a: estimate foveation params from a low-res UNWARPED image ──
        # Uses the shared detection backbone (real SAM3 once loaded) so the
        # estimate is driven by real encoder features at the paper's 512² — not
        # the warped image, since the warp depends on the params we predict here.
        small      = F.interpolate(image, (self.low_res_size, self.low_res_size),
                                   mode='bilinear', align_corners=False)
        feats_low  = self.backbone(small)                  # 3 FPN tensors (coarse→fine)
        # Global pool each scale and concatenate
        enc_vec    = torch.cat([f.flatten(2).mean(-1) for f in feats_low], dim=-1)  # (B, C*3)
        o, R       = self.fov_estimator(enc_vec)           # (B,2), (B,)

        # Baseline ablation: force R≈0 so w(r)=0 and Φ(x)=x everywhere.  This
        # keeps warp, embedding, and box-decode all consistent on the identity
        # transform (see train.py --no_foveation).
        if getattr(self.fov_estimator, "_no_foveation", False):
            R = torch.full_like(R, 1e-3)

        # ── Stage 1b: warp full-resolution image ──────────────────────────────
        warped     = self.warp_layer(image, o, R)          # (B, 3, H, W)

        # ── Stage 1c: hyperbolic embedding → augment queries ─────────────────
        fov_params  = torch.cat([o, R.unsqueeze(-1).expand(-1, 2)], dim=-1)  # (B,4)
        embedding   = self.hyperbolic_emb(fov_params)      # (B, query_dim)
        queries     = self.object_queries.weight.unsqueeze(0).expand(B, -1, -1)
        aug_queries = augment_queries(queries, embedding)  # (B, Q, D)

        # ── Stage 2a: encode warped image once, decode object queries ─────────
        features    = self.backbone(warped)                # 3-scale FPN features
        enc_ctx     = self.detr.encode(features)           # reused by denoising
        query_feats = self.detr.decode(enc_ctx, aug_queries)  # (B, Q, D)

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

        # ── DINO-style denoising: separate decoder pass on noised GT boxes ────
        # Run in its own pass (not concatenated) because the HF deformable
        # decoder has no query self-attention mask — concatenation would leak GT
        # info into the matching queries.
        dn_out = None
        if denoising is not None:
            dn_out = self._denoising_pass(enc_ctx, embedding, *denoising)

        if return_riemannian:
            if denoising is not None:
                return boxes_eu, class_logits, o, R, boxes_ri, dn_out
            return boxes_eu, class_logits, o, R, boxes_ri
        return boxes_eu, class_logits, o, R

    def _denoising_pass(self, enc_ctx, embedding, gt_boxes_list,
                        gt_labels_list, noise_scale: float = 0.4):
        """Build noised-GT queries and run one extra decoder pass.

        Each denoising query i for image b corresponds 1-to-1 to GT box i, so
        the denoising loss needs no Hungarian matching.  Returns a dict with
        padded predictions and a validity mask, or ``None`` if the batch has no
        GT boxes.

        Args:
            enc_ctx        : encoder context from ``self.detr.encode``
            embedding      : (B, query_dim) foveation embedding (added to queries)
            gt_boxes_list  : list of (M_b, 4) Euclidean GT boxes in [-1, 1]
            gt_labels_list : list of (M_b,) class indices
            noise_scale    : std of box-relative Gaussian noise
        """
        device = embedding.device
        B      = len(gt_boxes_list)
        counts = [int(g.shape[0]) for g in gt_boxes_list]
        M      = max(counts) if counts else 0
        if M == 0:
            return None

        dn_q    = embedding.new_zeros(B, M, self.query_dim)
        dn_ref  = embedding.new_full((B, M, 2), 0.5)
        dn_mask = torch.zeros(B, M, dtype=torch.bool, device=device)

        for b in range(B):
            m = counts[b]
            if m == 0:
                continue
            gt  = gt_boxes_list[b].to(device).float()      # (m,4) [cx,cy,w,h] in [-1,1]
            lbl = gt_labels_list[b].to(device)
            # Box-relative Gaussian noise on centre and size (DINO scheme).
            scale = torch.cat([gt[:, 2:3], gt[:, 3:4], gt[:, 2:3], gt[:, 3:4]], dim=-1)
            noisy = gt + torch.randn_like(gt) * noise_scale * scale
            centre = noisy[:, :2].clamp(-1.0, 1.0)
            # Content query = per-class label embedding + foveation context.
            dn_q[b, :m]   = self.dn_label_emb(lbl) + embedding[b].unsqueeze(0)
            dn_ref[b, :m] = ((centre + 1.0) * 0.5).clamp(0.0, 1.0)   # [-1,1] → [0,1]
            dn_mask[b, :m] = True

        dn_feats = self.detr.decode(enc_ctx, dn_q, reference_points=dn_ref)
        dn_boxes_ri, dn_logits = self.box_head(dn_feats)
        return {"boxes_ri": dn_boxes_ri, "logits": dn_logits, "mask": dn_mask}
