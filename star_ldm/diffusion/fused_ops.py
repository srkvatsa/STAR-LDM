"""Fused diffusion operations that minimize kernel launches.

Each function combines multiple elementwise operations on (B, D) tensors
into a single fused kernel via torch.jit.script.  On MPS / CUDA this
eliminates per-op dispatch overhead (significant when D=768, B=1).

When Metal compute shaders are available (MPS device), the Metal kernel
versions are used for lower dispatch overhead. Falls back to JIT-scripted
versions otherwise.
"""

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Metal kernel dispatch (lazy-loaded)
# ---------------------------------------------------------------------------

_metal_ddpm = None
_metal_ddpm_checked = False


def _get_metal_ddpm():
    """Lazy-load the Metal DDPM step kernel."""
    global _metal_ddpm, _metal_ddpm_checked
    if not _metal_ddpm_checked:
        _metal_ddpm_checked = True
        try:
            from star_ldm.kernels import get_ddpm_step_kernel
            _metal_ddpm = get_ddpm_step_kernel()
        except Exception:
            _metal_ddpm = None
    return _metal_ddpm


def metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda):
    """DDPM step via Metal compute shader. Returns None if not available."""
    mod = _get_metal_ddpm()
    if mod is not None and z_t.is_mps:
        try:
            return mod.ddpm_step(
                z_t.contiguous(), eps.contiguous(), noise.contiguous(),
                alpha2.contiguous(), alpha2_next.contiguous(), float(var_lambda),
            )
        except Exception:
            pass
    return None


@torch.jit.script
def _jit_fused_ddpm_step(
    z_t: Tensor,
    eps: Tensor,
    noise: Tensor,
    alpha2: Tensor,
    alpha2_next: Tensor,
    var_lambda: float,
) -> Tensor:
    """Fused DDPM denoising step.

    Replaces ~17 separate kernel launches with one fused op:
        alpha2_now = alpha2 / alpha2_next
        min_var = exp(log1p(-alpha2_next) - log1p(-alpha2)) * (1 - alpha2_now)
        max_var = 1 - alpha2_now
        sigma = exp(var_lambda * log(max_var) + (1-var_lambda) * log(min_var))
        z_{t-1} = 1/sqrt(alpha2_now) * (z_t - (1-alpha2_now)/sqrt(1-alpha2) * eps)
                  + sqrt(sigma) * noise

    Args:
        z_t: (B, D) current noisy embedding.
        eps: (B, D) predicted noise.
        noise: (B, D) fresh Gaussian noise.
        alpha2: (B, 1) current noise level.
        alpha2_next: (B, 1) next noise level.
        var_lambda: Variance interpolation parameter in [0, 1].

    Returns:
        z_{t-1}: (B, D) denoised embedding.
    """
    alpha2_now = alpha2 / alpha2_next
    # Variance bounds
    min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
    max_var = 1.0 - alpha2_now
    # Interpolated variance
    sigma = torch.exp(var_lambda * torch.log(max_var) + (1.0 - var_lambda) * torch.log(min_var))
    # Mean
    inv_sqrt_alpha2_now = 1.0 / torch.sqrt(alpha2_now)
    coeff = (1.0 - alpha2_now) / torch.sqrt(1.0 - alpha2)
    z_next = inv_sqrt_alpha2_now * (z_t - coeff * eps) + torch.sqrt(sigma) * noise
    return z_next


def fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda):
    """Fused DDPM step — JIT-scripted version.

    The Metal kernel is disabled: on (B,768) tensors the 9KB working set fits
    in L1 cache, so JIT fusion (0.040ms) is 3x faster than Metal dispatch
    (0.115ms) due to fixed kernel launch overhead.
    """
    return _jit_fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)


@torch.jit.script
def fused_ddim_step(
    x_start: Tensor,
    eps: Tensor,
    alpha2_next: Tensor,
) -> Tensor:
    """Fused DDIM denoising step.

    z_{t-1} = sqrt(alpha2_next) * x_start + sqrt(1 - alpha2_next) * eps

    Args:
        x_start: (B, D) predicted clean embedding.
        eps: (B, D) predicted noise.
        alpha2_next: (B, 1) next noise level.

    Returns:
        z_{t-1}: (B, D) denoised embedding.
    """
    return torch.sqrt(alpha2_next) * x_start + torch.sqrt(1.0 - alpha2_next) * eps


@torch.jit.script
def fused_v_to_x0_eps(
    z_t: Tensor,
    v: Tensor,
    alpha2: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused conversion from v-prediction to x_start and eps.

    x_start = sqrt(alpha2) * z_t - sqrt(1-alpha2) * v
    eps     = sqrt(alpha2) * v + sqrt(1-alpha2) * z_t  (not needed but cheap)

    Currently predict_start_from_v and predict_noise_from_v are separate calls
    in diff_utils.py, each doing their own sqrt. This computes both at once.

    Args:
        z_t: (B, D) noisy embedding.
        v: (B, D) v-prediction from model.
        alpha2: (B, 1) noise level.

    Returns:
        (x_start, eps): Both (B, D).
    """
    sqrt_alpha2 = torch.sqrt(alpha2)
    sqrt_one_minus_alpha2 = torch.sqrt(1.0 - alpha2)
    x_start = sqrt_alpha2 * z_t - sqrt_one_minus_alpha2 * v
    eps = sqrt_alpha2 * v + sqrt_one_minus_alpha2 * z_t
    return x_start, eps
