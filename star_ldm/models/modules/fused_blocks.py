"""Fused transformer building blocks for the 8-token micro-transformers.

Provides drop-in replacements for Attention and FeedForward from blocks.py
with fused operations that minimize kernel launches.  Key fusions:

1. **Fused RMSNorm + FiLM**: Combines L2-normalize → scale → gamma → FiLM
   modulation into a single fused op (1 kernel instead of 3-4).

2. **Fused 8-token attention**: Combines QK-norm + scaled dot product into a
   single operation optimized for seq_len=8. The 8×8 attention matrix per head
   fits entirely in registers — no tiling or online softmax needed.

Usage:
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks
    swap_to_fused_blocks(model.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model.score_net_head.transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

from star_ldm.models.modules.norm import RMSNorm


def exists(val):
    return val is not None


# ---------------------------------------------------------------------------
# Metal kernel dispatch (lazy-loaded)
# ---------------------------------------------------------------------------

_metal_rmsnorm = None
_metal_rmsnorm_checked = False
_metal_attn = None
_metal_attn_checked = False


def _get_metal_rmsnorm():
    global _metal_rmsnorm, _metal_rmsnorm_checked
    if not _metal_rmsnorm_checked:
        _metal_rmsnorm_checked = True
        try:
            from star_ldm.kernels import get_rmsnorm_film_kernel
            _metal_rmsnorm = get_rmsnorm_film_kernel()
        except Exception:
            _metal_rmsnorm = None
    return _metal_rmsnorm


def _get_metal_attn():
    global _metal_attn, _metal_attn_checked
    if not _metal_attn_checked:
        _metal_attn_checked = True
        try:
            from star_ldm.kernels import get_tiny_attention_kernel
            _metal_attn = get_tiny_attention_kernel()
        except Exception:
            _metal_attn = None
    return _metal_attn


def metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift):
    """RMSNorm+FiLM via Metal. Returns None if not available."""
    mod = _get_metal_rmsnorm()
    if mod is not None and x.is_mps:
        try:
            return mod.rmsnorm_film(
                x.contiguous(), gamma.contiguous(), float(dim_scale),
                film_scale, film_shift,
            )
        except Exception:
            pass
    return None


def metal_rmsnorm(x, gamma, dim_scale):
    """RMSNorm via Metal. Returns None if not available."""
    mod = _get_metal_rmsnorm()
    if mod is not None and x.is_mps:
        try:
            return mod.rmsnorm(x.contiguous(), gamma.contiguous(), float(dim_scale))
        except Exception:
            pass
    return None


def metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale):
    """Fused 8-token attention via Metal. Returns None if not available."""
    mod = _get_metal_attn()
    if mod is not None and q.is_mps and q.size(2) == 8:
        try:
            return mod.tiny_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                q_gamma.contiguous(), k_gamma.contiguous(),
                float(dim_head_scale), float(attn_scale),
            )
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Fused RMSNorm + FiLM (JIT fallback)
# ---------------------------------------------------------------------------

@torch.jit.script
def _jit_fused_rmsnorm_film(
    x: Tensor,
    gamma: Tensor,
    dim_scale: float,
    film_scale: Tensor,
    film_shift: Tensor,
) -> Tensor:
    """Fused RMSNorm + FiLM conditioning.

    Computes in one pass:
        x_norm = x / ||x||_2 * sqrt(dim) * gamma
        output = x_norm * (film_scale + 1) + film_shift

    Equivalent to 3-4 separate kernel launches in the unfused version.

    Args:
        x: (B, L, D) input tensor.
        gamma: (D,) learned RMSNorm scale.
        dim_scale: sqrt(dim), the RMSNorm dimension scale.
        film_scale: (B, 1, D) or (B, L, D) FiLM scale from time conditioning.
        film_shift: (B, 1, D) or (B, L, D) FiLM shift from time conditioning.

    Returns:
        (B, L, D) output tensor.
    """
    # L2 normalize along last dim
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    # Fuse: normalize * dim_scale * gamma * (film_scale + 1) + film_shift
    x_normed = x / norm * dim_scale
    return (x_normed * gamma) * (film_scale + 1.0) + film_shift


@torch.jit.script
def _jit_fused_rmsnorm(x: Tensor, gamma: Tensor, dim_scale: float) -> Tensor:
    """Fused RMSNorm without FiLM (for layers that don't use time conditioning)."""
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norm * dim_scale * gamma


def fused_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift):
    """RMSNorm+FiLM — Metal kernel on MPS, JIT fallback otherwise."""
    result = metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)
    if result is not None:
        return result
    return _jit_fused_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)


def fused_rmsnorm(x, gamma, dim_scale):
    """RMSNorm — Metal kernel on MPS, JIT fallback otherwise."""
    result = metal_rmsnorm(x, gamma, dim_scale)
    if result is not None:
        return result
    return _jit_fused_rmsnorm(x, gamma, dim_scale)


# ---------------------------------------------------------------------------
# Fused QK-Norm + Tiny Attention
# ---------------------------------------------------------------------------

@torch.jit.script
def _jit_fused_qknorm_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_gamma: Tensor,
    k_gamma: Tensor,
    dim_head_scale: float,
    attn_scale: float,
) -> Tensor:
    """Fused QK-norm + scaled dot-product attention for tiny sequences.

    For seq_len=8, the 8×8 attention matrix per head has only 64 values.
    This fuses QK-norm and attention into a single operation, avoiding
    the overhead of separate RMSNorm calls and F.scaled_dot_product_attention
    dispatch for such small inputs.

    Args:
        q: (B, H, S, D) query tensor.
        k: (B, H, S, D) key tensor.
        v: (B, H, S, D) value tensor.
        q_gamma: (D,) learned scale for Q norm.
        k_gamma: (D,) learned scale for K norm.
        dim_head_scale: sqrt(dim_head) for RMSNorm.
        attn_scale: 1/sqrt(dim_head) for attention scaling.

    Returns:
        (B, H, S, D) attention output.
    """
    # Fused QK-norm: normalize Q and K, apply learned scale
    q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    q = q / q_norm * dim_head_scale * q_gamma

    k_norm = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
    k = k / k_norm * dim_head_scale * k_gamma

    # Attention: QK^T / sqrt(d) -> softmax -> V
    # For 8 tokens, this is a tiny 8×8 matrix — fits in registers
    attn = torch.matmul(q, k.transpose(-2, -1)) * attn_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def fused_qknorm_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale):
    """QK-norm + tiny attention — Metal kernel on MPS, JIT fallback otherwise."""
    result = metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)
    if result is not None:
        return result
    return _jit_fused_qknorm_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)


# ---------------------------------------------------------------------------
# Fused Attention module (drop-in replacement)
# ---------------------------------------------------------------------------

class FusedAttention(nn.Module):
    """Drop-in replacement for Attention with fused RMSNorm+FiLM and fused
    QK-norm attention.
    """

    def __init__(self, original: 'Attention'):
        super().__init__()
        # Steal weights from original module
        self.heads = original.heads
        self.dropout = original.dropout
        self.causal = original.causal

        # RMSNorm params
        self.gamma = original.pre_norm.gamma
        self.dim_scale = original.pre_norm.scale

        # QKV projection
        self.to_qkv = original.to_qkv
        self.to_out = original.to_out

        # QK-norm params
        self.q_gamma = original.q_norm.gamma
        self.k_gamma = original.k_norm.gamma
        self.dim_head_scale = original.q_norm.scale
        dim_head = self.q_gamma.shape[0]
        self.attn_scale = 1.0 / math.sqrt(dim_head)

        # Time conditioning (FiLM)
        self.time_cond = getattr(original, 'time_cond', None)

    def forward(self, x, attn_bias, time_emb=None):
        batch, l, d = x.shape

        # Fused RMSNorm + FiLM
        if exists(time_emb) and exists(self.time_cond):
            film_params = self.time_cond(time_emb)
            film_scale, film_shift = film_params.chunk(2, dim=-1)
            x = fused_rmsnorm_film(x, self.gamma, self.dim_scale, film_scale, film_shift)
        else:
            x = fused_rmsnorm(x, self.gamma, self.dim_scale)

        # QKV projection (GEMM — can't fuse further without custom GEMM kernel)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Fused QK-norm + attention
        if exists(attn_bias) or self.causal:
            # Fall back to standard path for attn_bias/causal (rare in micro-transformers)
            q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
            q = q / q_norm * self.dim_head_scale * self.q_gamma
            k_norm = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
            k = k / k_norm * self.dim_head_scale * self.k_gamma

            if exists(attn_bias):
                attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, self.heads, -1, -1)
                if self.causal:
                    from star_ldm.models.modules.blocks import create_causal_mask
                    q_len, k_len = q.shape[-2], k.shape[-2]
                    mask_value = -torch.finfo(q.dtype).max
                    causal_mask = create_causal_mask(q_len, k_len, device=q.device)
                    attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)

            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.
            )
        else:
            # Fast path: fused QK-norm + tiny attention (no mask, non-causal)
            out = fused_qknorm_attention(
                q, k, v,
                self.q_gamma, self.k_gamma,
                self.dim_head_scale, self.attn_scale,
            )

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


# ---------------------------------------------------------------------------
# Fused FeedForward module (drop-in replacement)
# ---------------------------------------------------------------------------

class FusedFeedForward(nn.Module):
    """Drop-in replacement for FeedForward with fused RMSNorm+FiLM."""

    def __init__(self, original: 'FeedForward'):
        super().__init__()
        # RMSNorm params
        self.gamma = original.pre_norm.gamma
        self.dim_scale = original.pre_norm.scale

        # Time conditioning
        self.time_cond = original.time_cond

        # FFN layers (GLU + dropout + linear)
        self.net = original.net

    def forward(self, x, time_emb=None):
        # Fused RMSNorm + FiLM
        if exists(self.time_cond) and exists(time_emb):
            film_params = self.time_cond(time_emb)
            film_scale, film_shift = film_params.chunk(2, dim=-1)
            x = fused_rmsnorm_film(x, self.gamma, self.dim_scale, film_scale, film_shift)
        else:
            x = fused_rmsnorm(x, self.gamma, self.dim_scale)

        x = self.net(x)
        return x


# ---------------------------------------------------------------------------
# Module swap utility
# ---------------------------------------------------------------------------

def swap_to_fused_blocks(transformer_model: nn.Module):
    """Replace Attention and FeedForward modules with fused versions in-place.

    Works on TransformerModel by iterating over its TransformerBlocks.

    Args:
        transformer_model: A TransformerModel instance (or any module
            containing TransformerBlocks with .attn and .ff attributes).
    """
    from star_ldm.models.modules.blocks import Attention, FeedForward

    for block in transformer_model.modules():
        if hasattr(block, 'attn') and isinstance(block.attn, Attention):
            block.attn = FusedAttention(block.attn)
        if hasattr(block, 'ff') and isinstance(block.ff, FeedForward):
            block.ff = FusedFeedForward(block.ff)

    return transformer_model
