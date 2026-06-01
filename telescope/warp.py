"""
telescope.warp
==============
Differentiable image warp using the hyperbolic foveated transform.

The warp implements inverse mapping:  I'(y) = I(Φ⁻¹(y))

For each output pixel position y, we ask "which input pixel fills this?"
The answer is x = Φ⁻¹(y), computed via Newton-Raphson.
PyTorch's grid_sample then performs bilinear interpolation at those positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .geometry import hyperbolic_inverse

__all__ = ["create_output_grid", "compute_source_grid", "FoveationWarpLayer"]


def create_output_grid(height: int, width: int, device=None) -> Tensor:
    """Uniform normalised grid covering [-1, 1]² for an (H, W) image.

    Returns (H, W, 2) where [..., 0] = x (horizontal) and [..., 1] = y (vertical).
    Suitable to pass directly to compute_source_grid or F.grid_sample.

    Args:
        height, width : output image dimensions
    """
    xs = torch.linspace(-1.0, 1.0, width,  device=device)
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each
    return torch.stack([grid_x, grid_y], dim=-1)              # (H, W, 2)


def compute_source_grid(
    height: int,
    width: int,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
    device=None,
) -> Tensor:
    """For each output pixel y, compute x = Φ⁻¹(y) — where to sample in the input.

    Handles both single-image and batched calls:
        single  :  o=(2,),  R=scalar  → returns (1, H, W, 2)
        batched :  o=(B,2), R=(B,)    → returns (B, H, W, 2)

    The NR runs fully vectorised over all H×W pixels simultaneously.
    We loop over the batch dimension (typically B=4), not the pixel dimension,
    so the memory footprint stays at O(H×W) per image.
    """
    output_grid = create_output_grid(height, width, device=device)   # (H, W, 2)
    grid_flat   = output_grid.view(-1, 2)                            # (H*W, 2)

    if o.dim() == 1:
        o = o.unsqueeze(0)    # (1, 2)
        R = R.unsqueeze(0)    # (1,)

    source_maps = []
    for b in range(o.shape[0]):
        x_flat = hyperbolic_inverse(grid_flat, o[b], R[b], alpha, p)   # (H*W, 2)
        source_maps.append(x_flat.view(height, width, 2))

    return torch.stack(source_maps, dim=0)   # (B, H, W, 2)


class FoveationWarpLayer(nn.Module):
    """Differentiable hyperbolic foveation warp.

    Applies  I'(y) = I(Φ⁻¹(y))  to an image tensor using F.grid_sample.

    In plain terms: stretches the region near (o, R) so distant objects
    appear larger in the output, while peripheral regions are compressed.
    This is the re-sampling layer described in §3.1 of the Telescope paper.

    Usage::

        warp = FoveationWarpLayer(alpha=2.0, p=2.0)
        warped_image = warp(image, o, R)    # (B, C, H, W)

    Args:
        alpha        : Hyperbolic contraction strength. Paper default: 2.0
        p            : Blending exponent.               Paper default: 2.0
        mode         : Interpolation — 'bilinear' (smooth, differentiable)
                       or 'nearest' (fast but not differentiable).
        padding_mode : What to return for out-of-bounds source coords.
                       'border' repeats edge pixels — avoids black borders.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        p: float = 2.0,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        self.alpha        = alpha
        self.p            = p
        self.mode         = mode
        self.padding_mode = padding_mode

    def forward(self, image: Tensor, o: Tensor, R: Tensor) -> Tensor:
        """Warp image(s) using hyperbolic foveation.

        Args:
            image : (B, C, H, W) float tensor
            o     : (B, 2) or (2,) foveation centre in [-1, 1]²
            R     : (B,)   or ()   radial scale > 0
        Returns:
            (B, C, H, W) warped image — same shape and dtype as input
        """
        B, C, H, W = image.shape

        if o.dim() == 1:
            o = o.unsqueeze(0).expand(B, -1)
        if R.dim() == 0:
            R = R.unsqueeze(0).expand(B)

        # Newton–Raphson Φ⁻¹ uses EPS=1e-8, which underflows in fp16; compute the
        # sampling grid in fp32 (autocast off) then cast to the image dtype.
        with torch.autocast(device_type=image.device.type, enabled=False):
            source_grid = compute_source_grid(
                H, W, o.float(), R.float(), self.alpha, self.p, device=image.device
            )   # (B, H, W, 2)

        return F.grid_sample(
            image,
            source_grid.to(image.dtype),
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )
