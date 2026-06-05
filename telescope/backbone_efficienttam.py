"""
telescope.backbone_efficienttam
===============================
Lightweight **EfficientTAM** image encoder (frozen) as a drop-in replacement for
:class:`telescope.backbone_sam3.SAM3Backbone` — same feature contract, ~10-40×
fewer parameters, edge-real-time friendly (see EfficientTAM: ~151 FPS on A100,
>10 FPS on an iPhone, vs SAM 3.1's ~30 ms only on an H200).

Why this fits Telescope's pipeline
-----------------------------------
``TelescopeModel`` calls the backbone with two hard requirements (pipeline.py):
  1. ``FoveationEstimator(query_dim * 3)`` consumes ``cat([pool(f) for f in feats])``
     → the backbone must return **exactly 3 feature maps of 256 channels each**.
  2. ``RealDeformableDetr`` uses ``features[:2]`` (the two coarsest) at 256 ch.

EfficientTAM's image encoder is a **plain ViT trunk + a single-scale ViTDet neck**
(see efficient_track_anything/.../image_encoder.py: ``ViTDetNeck.forward`` returns
only ``out[0]``).  So it emits **one** 256-channel map at stride ~16.  We wrap it
with a tiny parameter-free feature pyramid that produces the 3 coarse→fine levels
the pipeline expects.  Keeping the pyramid parameter-free means the whole backbone
stays frozen (matching how train.py freezes ``model.backbone``), and the detector's
trainable per-level ``input_proj`` does the per-scale adaptation.

The foveation is what makes a coarse-but-fast backbone viable for tiny drones: the
warp magnifies the target region, so a 5-px drone lands on several feature cells
even at stride 16.

Install (offline server, like sam3):
    pip install -e EfficientTAM --no-index --no-build-isolation --no-deps
    # weights: efficienttam_s.pt from https://huggingface.co/yunyangx/efficient-track-anything
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["EfficientTAMBackbone"]

# SAM2/EfficientTAM image normalisation (ImageNet mean/std on [0,1] inputs).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Default config + checkpoint (the "Small" 1024² variant — best accuracy ceiling;
# switch to *_512x512 for a faster edge variant).
_DEFAULT_CFG = "configs/efficienttam/efficienttam_s.yaml"


class EfficientTAMBackbone(nn.Module):
    """Frozen EfficientTAM ViT image encoder, exposing SAM3Backbone's contract.

    Args:
        checkpoint_path : path to ``efficienttam_s.pt`` (or ``None`` for random
                          init — only useful for shape tests).
        out_channels    : FPN channel dim.  Must be 256 (the ViTDet neck d_model).
        config_file     : EfficientTAM Hydra config name (selects the variant).
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        out_channels: int = 256,
        config_file: str = _DEFAULT_CFG,
    ) -> None:
        super().__init__()
        if out_channels != 256:
            raise ValueError(
                f"EfficientTAM ViTDet neck d_model is fixed at 256, got "
                f"out_channels={out_channels}.  Run TelescopeModel with query_dim=256."
            )
        self.out_channels = out_channels

        # Build ONLY the image encoder (trunk + neck), not the mask decoder /
        # memory modules — via Hydra, using EfficientTAM's own config so the
        # architecture matches the checkpoint exactly across variants.
        import efficient_track_anything                 # registers the Hydra config path
        from hydra import compose
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg = compose(config_name=config_file)
        OmegaConf.resolve(cfg)
        self.image_encoder = instantiate(cfg.model.image_encoder, _recursive_=True)

        if checkpoint_path is not None:
            self._load_encoder_weights(checkpoint_path)

        # Frozen feature extraction: eval mode keeps drop-path / any train-only
        # behaviour off (the ViT here is small and fp16/bf16-stable).
        self.image_encoder.eval()

        # Normalisation buffers (broadcast over B,H,W); move with .to(device).
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval even when the parent calls .train()."""
        super().train(mode)
        self.image_encoder.eval()
        return self

    # ── checkpoint loading ────────────────────────────────────────────────────
    def _load_encoder_weights(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = ckpt["model"] if "model" in ckpt else ckpt

        prefix = "image_encoder."
        enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not enc_sd:
            raise RuntimeError(
                f"No '{prefix}*' weights found in {checkpoint_path}. "
                "Is this an EfficientTAM checkpoint?"
            )
        missing, unexpected = self.image_encoder.load_state_dict(enc_sd, strict=False)
        # rel-pos / pos-embed buffers may be interpolated lazily; only real
        # missing weights are a problem.
        real_missing = [k for k in missing if "pos" not in k.lower()]
        if real_missing:
            raise RuntimeError(
                f"EfficientTAM encoder missing {len(real_missing)} expected weights, "
                f"e.g. {real_missing[:5]}. Wrong checkpoint or config variant?"
            )
        print(f"[backbone] loaded {len(enc_sd) - len(unexpected)} EfficientTAM "
              f"encoder weights (ignored {len(unexpected)} extra keys)")

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, image: Tensor):
        """
        Args:
            image : (B, 3, H, W) in [0, 1]
        Returns:
            list of 3 tensors, coarse→fine, each (B, 256, H_i, W_i).  At a 1024
            input the native ViT map is 64×64 (stride 16); we emit
            [32², 64², 128²].  RealDeformableDetr uses the two coarsest
            ([32², 64²]); the finest only feeds the FoveationEstimator's pooled
            vector, which is resolution-agnostic.
        """
        x = (image - self._mean) / self._std

        # Frozen ⇒ run under no_grad (the only gradient path out of a frozen
        # backbone reaches Φ⁻¹, which passes no grad to o,R, and the fixed input
        # image — i.e. no trainable parameter).  bf16 for speed/stability.
        frozen = not any(p.requires_grad for p in self.image_encoder.parameters())
        grad_ctx = torch.no_grad() if frozen else contextlib.nullcontext()
        amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if image.is_cuda else contextlib.nullcontext())
        with grad_ctx, amp_ctx:
            out = self.image_encoder(x)
            f = out["backbone_fpn"][-1]          # (B, 256, Hf, Wf) single scale

        f = f.to(image.dtype)
        Hf, Wf = f.shape[-2:]
        coarse = F.interpolate(f, (max(1, Hf // 2), max(1, Wf // 2)),
                               mode="bilinear", align_corners=False)
        fine   = F.interpolate(f, (Hf * 2, Wf * 2),
                               mode="bilinear", align_corners=False)
        return [coarse, f, fine]                 # coarse→fine
