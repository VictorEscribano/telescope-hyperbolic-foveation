"""
telescope.geometry
==================
Core mathematical primitives for the hyperbolic foveated transform.

Public API
----------
poincare_projection          — h(x; o)         Eq.(1)
blend_weight                 — w(r)
hyperbolic_foveated_transform — Φ(x)            Eq.(2)
compute_jacobian             — J_Φ(x)
HyperbolicInverseNR          — custom autograd for Φ⁻¹
hyperbolic_inverse           — functional wrapper for Φ⁻¹  Eq.(3)
validate_inversion           — round-trip error report
"""

import warnings
import torch
from torch import Tensor

__all__ = [
    "EPS",
    "poincare_projection",
    "blend_weight",
    "hyperbolic_foveated_transform",
    "compute_jacobian",
    "HyperbolicInverseNR",
    "hyperbolic_inverse",
    "validate_inversion",
]

EPS: float = 1e-8


# ── Poincaré projection ───────────────────────────────────────────────────────

def poincare_projection(x: Tensor, o: Tensor, alpha: float) -> Tensor:
    """h(x; o) = o + tanh(α·r)/r · (x − o).   Eq.(1).

    Radially contracts coordinates toward o.  At r=0 the limit (via
    L'Hôpital) gives scale = α, so the transform is locally linear near o.

    Args:
        x     : (..., 2) image coordinates in [-1, 1]²
        o     : (2,) foveation centre
        alpha : hyperbolic contraction strength > 0
    Returns:
        (..., 2)
    """
    d = x - o
    r_safe = torch.norm(d, dim=-1, keepdim=True).clamp(min=EPS)
    scale = torch.tanh(alpha * r_safe) / r_safe
    return o + scale * d


# ── Blend weight ──────────────────────────────────────────────────────────────

def blend_weight(r: Tensor, R: Tensor, p: float) -> Tensor:
    """w(r) = (1 − clamp(r/R, 0, 1))^p.

    w(0) = 1  → full Poincaré projection at the foveation centre.
    w(R) = 0  → identity transform at and beyond radius R.

    Args:
        r : (..., 1) radial distances
        R : scalar tensor, radial scale > 0
        p : blending exponent > 0
    Returns:
        (..., 1) weights in [0, 1]
    """
    r_over_R = (r / R.clamp(min=EPS)).clamp(max=1.0)
    return (1.0 - r_over_R) ** p


# ── Forward transform Φ ───────────────────────────────────────────────────────

def hyperbolic_foveated_transform(
    x: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
) -> Tensor:
    """Φ(x) = (1 − w(r))·x + w(r)·h(x; o).   Eq.(2).

    For r << R : magnifies the region around foveation centre o.
    For r >= R : identity (no warping outside the Poincaré radius).

    Paper defaults: alpha=2.0, p=2.0 (found via grid search on TruckDrive).

    Args:
        x     : (..., 2) normalised image coordinates
        o     : (2,) foveation centre (predicted by FoveationEstimator)
        R     : scalar tensor, radial scale   (predicted by FoveationEstimator)
        alpha : hyperbolic contraction strength  [paper: 2.0]
        p     : blending exponent               [paper: 2.0]
    Returns:
        (..., 2) transformed coordinates
    """
    r = torch.norm(x - o, dim=-1, keepdim=True).clamp(min=EPS)
    h = poincare_projection(x, o, alpha)
    w = blend_weight(r, R, p)
    return (1.0 - w) * x + w * h


# ── Analytical Jacobian J_Φ ───────────────────────────────────────────────────

def compute_jacobian(
    x: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
) -> Tensor:
    """Analytical Jacobian J_Φ(x) ∈ R^(2×2) for each point.

    Derived from Φ(x) = x + f(r)·d  where  f(r) = w(r)·(s(r)−1):

        J_Φ = (1 + f(r))·I₂  +  (f'(r)/r)·(d ⊗ d)

    This rank-1 structure has two eigenvalues:
        λ_tangential = 1 + f(r)
        λ_radial     = 1 + f(r) + r·f'(r)

    Args:
        x : (..., 2)
        o : (2,)
        R : scalar tensor
    Returns:
        (..., 2, 2)
    """
    d = x - o
    r = torch.norm(d, dim=-1, keepdim=True).clamp(min=EPS)

    # s(r) = tanh(αr)/r  and  s'(r)
    ar      = alpha * r
    tanh_ar = torch.tanh(ar)
    s       = tanh_ar / r
    ds_dr   = alpha * (1.0 - tanh_ar.pow(2)) / r - s / r

    # w(r) and dw/dr
    r_over_R  = (r / R.clamp(min=EPS)).clamp(max=1.0)
    one_minus = 1.0 - r_over_R
    w         = one_minus ** p
    inside_R  = r < R
    om_safe   = torch.where(inside_R, one_minus, torch.ones_like(one_minus))
    dw_dr     = torch.where(
        inside_R,
        -p / R.clamp(min=EPS) * om_safe.pow(p - 1),
        torch.zeros_like(r),
    )

    # f(r) = w·(s−1),  f'(r) = w'·(s−1) + w·s'
    f     = w * (s - 1.0)
    df_dr = dw_dr * (s - 1.0) + w * ds_dr

    # J = (1+f)·I₂ + (f'/r)·(d⊗d)
    I2      = torch.eye(2, dtype=x.dtype, device=x.device)
    I2      = I2.expand(*d.shape[:-1], 2, 2).clone()
    d_outer = d.unsqueeze(-1) * d.unsqueeze(-2)

    return (1.0 + f).unsqueeze(-1) * I2 + (df_dr / r).unsqueeze(-1) * d_outer


# ── Newton-Raphson inverse Φ⁻¹  ──────────────────────────────────────────────

class HyperbolicInverseNR(torch.autograd.Function):
    """Differentiable Φ⁻¹ via Newton-Raphson + implicit-function-theorem backward.

    Forward  : Newton iteration x^{k+1} = x^k + η·J_Φ(x^k)^{−1}·(y − Φ(x^k))
               inside no_grad().  η=1 is the full Newton step (quadratic
               convergence); η<1 under-relaxes if a region is ill-conditioned.
    Backward : dL/dy = J_Φ(x*)^{−T}·dL/dx*, and (via the implicit function
               theorem) dL/do = −Φ_o^T·g, dL/dR = −Φ_R^T·g with g = dL/dy —
               so gradients flow to the foveation params, not only to y.

    This is the "fixed-point differentiation" trick used in Deep Equilibrium
    Models: the backward is exact regardless of how many NR steps the forward took.
    """

    @staticmethod
    def forward(
        ctx,
        y: Tensor,
        o: Tensor,
        R: Tensor,
        alpha: float,
        p: float,
        eta: float,
        max_iter: int,
        tol: float,
    ) -> Tensor:
        with torch.no_grad():
            x = y.clone()
            for _ in range(max_iter):
                residual = y - hyperbolic_foveated_transform(x, o, R, alpha, p)
                if residual.abs().max().item() < tol:
                    break
                # Newton step: solve J_Φ(x)·Δx = residual  (closed-form 2×2),
                # the same Jacobian solve the backward uses.  Quadratic
                # convergence — replaces the old fixed-step Picard iteration
                # x += η·residual, which stalled wherever ‖I − η·J_Φ‖ ≮ 1.
                J = compute_jacobian(x, o, R, alpha, p)
                delta = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
                x = (x + eta * delta).clamp(-1.5, 1.5)
            else:
                # Re-evaluate after the final step so the report is accurate;
                # only warn if it is genuinely still above tolerance.
                residual = y - hyperbolic_foveated_transform(x, o, R, alpha, p)
                if residual.abs().max().item() >= tol:
                    warnings.warn(
                        f"Newton did not converge in {max_iter} iters. "
                        f"Max residual: {residual.abs().max():.2e} (tol={tol:.2e})",
                        RuntimeWarning,
                    )
        ctx.save_for_backward(x, o, R)
        ctx.alpha, ctx.p = alpha, p
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_star, o, R = ctx.saved_tensors
        alpha, p = ctx.alpha, ctx.p

        # Implicit-function-theorem backward on the fixed point  y = Φ(x*; o, R).
        # Differentiating  J_x·dx* + Φ_o·do + Φ_R·dR = dy  and writing
        # g := J_x^{-T}·(dL/dx*)  gives the three vector-Jacobian products:
        #     dL/dy = g,   dL/do = -Φ_o^T·g,   dL/dR = -Φ_R^T·g
        # so the inverse is differentiable w.r.t. the foveation params (o, R),
        # not only y (Telescope paper §3.1: "the inverse ... admits
        # backpropagation through foveation parameters (o, R)").
        J = compute_jacobian(x_star, o, R, alpha, p)
        g = torch.linalg.solve(J.mT, grad_output.unsqueeze(-1)).squeeze(-1)   # = dL/dy

        grad_o = grad_R = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            # -Φ_o^T·g and -Φ_R^T·g via a single autograd VJP through Φ evaluated
            # at the fixed point (x* held constant).  Cheap: one forward of Φ.
            with torch.enable_grad():
                o_ = o.detach().requires_grad_(True)
                R_ = R.detach().requires_grad_(True)
                y_pred = hyperbolic_foveated_transform(x_star.detach(), o_, R_, alpha, p)
                grad_o, grad_R = torch.autograd.grad(
                    y_pred, (o_, R_), grad_outputs=-g, allow_unused=True,
                )
            if not ctx.needs_input_grad[1]:
                grad_o = None
            if not ctx.needs_input_grad[2]:
                grad_R = None

        return g, grad_o, grad_R, None, None, None, None, None


def hyperbolic_inverse(
    y: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
    eta: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tensor:
    """Functional interface: x* = Φ⁻¹(y).   Eq.(3).

    Solved by Newton-Raphson (full step η=1 → quadratic convergence).

    Differentiable — gradients flow via the implicit function theorem.
    """
    return HyperbolicInverseNR.apply(y, o, R, alpha, p, eta, max_iter, tol)


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_inversion(
    points: Tensor,
    o: Tensor,
    R: Tensor,
    alpha: float = 2.0,
    p: float = 2.0,
    warn_tol: float = 1e-5,
) -> dict:
    """Measure round-trip error ‖P − Φ⁻¹(Φ(P))‖ and print a report.

    Raises UserWarning if any error exceeds warn_tol.

    Args:
        points   : (N, 2) or (2,) input points in [-1, 1]²
        warn_tol : threshold above which a warning is emitted  [default: 1e-5]
    Returns:
        dict(max_error, mean_error, errors_per_point, all_passed)
    """
    if points.dim() == 1:
        points = points.unsqueeze(0)

    with torch.no_grad():
        phi_p   = hyperbolic_foveated_transform(points, o, R, alpha, p)
        recon   = hyperbolic_inverse(phi_p, o, R, alpha, p)
        errors  = torch.norm(points - recon, dim=-1)

    max_err  = errors.max().item()
    mean_err = errors.mean().item()
    passed   = max_err < warn_tol

    sep = "=" * 58
    print(sep)
    print("INVERSION VALIDATION REPORT")
    print(sep)
    print(f"  Points      : {points.shape[0]}")
    print(f"  Params      : o={[round(v, 3) for v in o.tolist()]}  "
          f"R={R.item():.3f}  alpha={alpha}  p={p}")
    print(f"  Max error   : {max_err:.2e}")
    print(f"  Mean error  : {mean_err:.2e}")
    print(f"  Tolerance   : {warn_tol:.2e}")
    if passed:
        print("  Status      : PASS")
    else:
        bad = (errors >= warn_tol).nonzero(as_tuple=True)[0]
        print(f"  Status      : WARNING — {len(bad)} point(s) exceed tolerance")
        warnings.warn(
            f"Inversion error {max_err:.2e} > tolerance {warn_tol:.2e}",
            UserWarning,
        )
    print(sep)

    return dict(
        max_error=max_err,
        mean_error=mean_err,
        errors_per_point=errors,
        all_passed=passed,
    )
