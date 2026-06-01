"""
telescope.backbone_sam3
=======================
Real SAM 3.1 vision backbone (frozen) as a drop-in replacement for
``SAM3EncoderStub``.

The Telescope detection path expects a backbone that maps an image to a list
of ``query_dim``-channel FPN feature maps, ordered **coarse → fine** (the same
contract as :class:`telescope.pipeline.SAM3EncoderStub`, whose
``[f_{H/4}, f_{H/2}, f_{H}]`` is coarsest-first).  ``RealDeformableDetr`` then
consumes the two coarsest levels (``features[:2]``).

This module builds only the *visual* half of SAM 3.1 — the ViT trunk plus the
SimpleFPN ("ViTDet") neck — and loads just those weights from the multiplex
checkpoint.  The text encoder, fusion transformer, segmentation head, and
tracker are not built, so memory stays close to the ~454 M-parameter backbone.

Geometry of the SAM 3.1 neck (``img_size=1008``, ``patch_size=14`` → 72×72 ViT
grid), with ``scale_factors=(4.0, 2.0, 1.0)`` the neck emits three 256-channel
maps at 288², 144², 72² (strides ≈ 3.5 / 7 / 14).  We drop the 0.5× level
(36²) — it carries no trained weights in the released checkpoint (SAM 3 applies
``scalp=1``) — and return the remaining three reversed to coarse→fine.

The backbone is frozen at train time.  Its gradient w.r.t. the warped input
would only flow back into Φ⁻¹ (which does not pass gradient to o, R) and the
fixed input image, so it reaches no trainable parameter; we therefore run the
trunk under ``no_grad`` when frozen and skip storing the ViT's activations,
which is what keeps training within a single 14–24 GB GPU.
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SAM3Backbone"]

# SAM 3.1 image pre-processing (see sam3/model/sam3_image_processor.py):
#   Resize(1008) → Normalize(mean=0.5, std=0.5)  ⇒  x ↦ 2x − 1 on [0,1] inputs.
_SAM3_RESOLUTION = 1008
_VISION_PREFIX   = "detector.backbone.vision_backbone."


class SAM3Backbone(nn.Module):
    """Frozen SAM 3.1 ViT + SimpleFPN neck, exposing the stub's feature contract.

    Args:
        checkpoint_path : path to ``sam3.1_multiplex.pt`` (or ``None`` for
                          random init — only useful for shape tests).
        out_channels    : FPN channel dim.  Must be 256 (the SAM 3 neck d_model).
        resolution      : square resolution the ViT runs at.  Default 1008.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        out_channels: int = 256,
        resolution: int = _SAM3_RESOLUTION,
    ) -> None:
        super().__init__()
        if out_channels != 256:
            raise ValueError(
                f"SAM3 neck d_model is fixed at 256, got out_channels={out_channels}. "
                "Run TelescopeModel with query_dim=256 to use the real backbone."
            )

        # Import lazily so the package still imports without the sam3 repo present.
        from sam3.model_builder import _create_vit_backbone, _create_position_encoding
        from sam3.model.necks import Sam3DualViTDetNeck

        self.resolution   = resolution
        self.out_channels = out_channels

        position_encoding = _create_position_encoding(precompute_resolution=resolution)
        vit_backbone      = _create_vit_backbone()
        # 3 levels (4×, 2×, 1×); the 0.5× level has no trained weights in the
        # released checkpoint, so we don't build it.
        self.neck = Sam3DualViTDetNeck(
            trunk=vit_backbone,
            position_encoding=position_encoding,
            d_model=out_channels,
            scale_factors=(4.0, 2.0, 1.0),
            add_sam2_neck=False,
        )

        if checkpoint_path is not None:
            self._load_vision_weights(checkpoint_path)

        # The backbone is frozen feature extraction: keep it in eval mode so the
        # ViT's training-only activation checkpointing and drop-path stay off.
        self.neck.eval()

    def train(self, mode: bool = True):
        """Keep the frozen backbone in eval even when the parent calls .train().

        The SAM3 ViT enables activation checkpointing and stochastic drop-path
        only in training mode; both are wrong for a frozen, no-grad backbone.
        """
        super().train(mode)
        self.neck.eval()
        return self

    # ── checkpoint loading ────────────────────────────────────────────────────
    def _load_vision_weights(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        vision_sd = {
            k[len(_VISION_PREFIX):]: v
            for k, v in ckpt.items()
            if k.startswith(_VISION_PREFIX)
        }
        missing, unexpected = self.neck.load_state_dict(vision_sd, strict=False)

        # Unexpected keys are the SAM2/interactive neck heads we deliberately
        # don't build; they are expected.  Missing keys would mean our trunk/neck
        # diverged from the checkpoint — surface those loudly.
        real_missing = [k for k in missing if not k.endswith("relative_coords")]
        if real_missing:
            raise RuntimeError(
                f"SAM3 backbone checkpoint is missing {len(real_missing)} expected "
                f"weights, e.g. {real_missing[:5]}. Wrong checkpoint or sam3 version?"
            )
        n_loaded = len(vision_sd) - len(unexpected)
        print(f"[backbone] loaded {n_loaded} SAM3 vision weights "
              f"(ignored {len(unexpected)} interactive/sam2 keys)")

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, image: Tensor):
        """
        Args:
            image : (B, 3, H, W) in [0, 1]
        Returns:
            list of 3 tensors, coarse→fine: [(B,256,72,72), (B,256,144,144),
            (B,256,288,288)] at the default 1008 resolution.  Matches the
            SAM3EncoderStub contract so RealDeformableDetr (features[:2]) works
            unchanged.
        """
        x = F.interpolate(
            image, (self.resolution, self.resolution),
            mode="bilinear", align_corners=False,
        )
        x = x * 2.0 - 1.0   # SAM3 Normalize(mean=0.5, std=0.5)

        # Run the backbone in its native bf16 regardless of the caller's autocast
        # dtype: the SAM3 ViT is numerically unstable in fp16 (produces NaNs),
        # bf16 is what it was trained in.  Frozen ⇒ no_grad, so we skip the ViT
        # activation graph entirely (the only gradient path out of the backbone
        # reaches Φ⁻¹, which passes no grad to o,R, and the fixed input image —
        # i.e. no trainable parameter).
        frozen = not any(p.requires_grad for p in self.neck.parameters())
        grad_ctx = torch.no_grad() if frozen else contextlib.nullcontext()
        with grad_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            sam3_features, _sam3_pos, _s2, _s2p = self.neck(x)

        # Neck emits fine→coarse [288², 144², 72²]; reverse to coarse→fine and
        # cast back to the pipeline's dtype so the (fp16/fp32) DETR matches.
        return [f.to(image.dtype) for f in sam3_features[::-1]]
