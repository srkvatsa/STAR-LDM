"""Correctness tests for Metal compute shaders.

Each Metal kernel is compared against the JIT-scripted reference
implementation. Maximum absolute difference must be < 1e-5.

These tests require MPS (Apple Silicon). They are skipped on other platforms.
"""

import pytest
import torch
import math

# Skip all tests if MPS is not available
pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
    reason="MPS not available"
)


# ---- DDPM Step Kernel ----

class TestMetalDDPMStep:
    """Compare Metal DDPM step kernel against JIT-scripted version."""

    @pytest.fixture
    def tensors(self):
        """Create test tensors on MPS."""
        B, D = 4, 768
        z_t = torch.randn(B, D, device='mps')
        eps = torch.randn(B, D, device='mps')
        noise = torch.randn(B, D, device='mps')
        alpha2 = torch.rand(B, 1, device='mps') * 0.8 + 0.1  # (0.1, 0.9)
        alpha2_next = alpha2 * torch.rand(B, 1, device='mps') * 0.5 + 0.3
        var_lambda = 0.2
        return z_t, eps, noise, alpha2, alpha2_next, var_lambda

    def test_ddpm_step_correctness(self, tensors):
        """Metal DDPM step matches JIT version within tolerance."""
        from star_ldm.diffusion.fused_ops import _jit_fused_ddpm_step, metal_ddpm_step

        z_t, eps, noise, alpha2, alpha2_next, var_lambda = tensors

        ref = _jit_fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)
        metal_result = metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)

        if metal_result is None:
            pytest.skip("Metal DDPM kernel not compiled")

        torch.mps.synchronize()
        max_diff = (ref - metal_result).abs().max().item()
        assert max_diff < 1e-5, f"DDPM step max diff: {max_diff}"

    def test_ddpm_step_batch_sizes(self):
        """Test various batch sizes."""
        from star_ldm.diffusion.fused_ops import _jit_fused_ddpm_step, metal_ddpm_step

        for B in [1, 2, 4, 8]:
            D = 768
            z_t = torch.randn(B, D, device='mps')
            eps = torch.randn(B, D, device='mps')
            noise = torch.randn(B, D, device='mps')
            alpha2 = torch.rand(B, 1, device='mps') * 0.8 + 0.1
            alpha2_next = alpha2 * 0.5

            ref = _jit_fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)
            result = metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)

            if result is None:
                pytest.skip("Metal DDPM kernel not compiled")

            torch.mps.synchronize()
            max_diff = (ref - result).abs().max().item()
            assert max_diff < 1e-5, f"B={B}: max diff {max_diff}"


# ---- RMSNorm + FiLM Kernel ----

class TestMetalRMSNormFiLM:
    """Compare Metal RMSNorm+FiLM kernel against JIT-scripted version."""

    @pytest.fixture
    def tensors(self):
        B, L, D = 2, 8, 768
        x = torch.randn(B, L, D, device='mps')
        gamma = torch.randn(D, device='mps')
        dim_scale = math.sqrt(D)
        film_scale = torch.randn(B, 1, D, device='mps')
        film_shift = torch.randn(B, 1, D, device='mps')
        return x, gamma, dim_scale, film_scale, film_shift

    def test_rmsnorm_film_correctness(self, tensors):
        """Metal RMSNorm+FiLM matches JIT version."""
        from star_ldm.models.modules.fused_blocks import (
            _jit_fused_rmsnorm_film, metal_rmsnorm_film,
        )

        x, gamma, dim_scale, film_scale, film_shift = tensors

        ref = _jit_fused_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)
        result = metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)

        if result is None:
            pytest.skip("Metal RMSNorm+FiLM kernel not compiled")

        torch.mps.synchronize()
        max_diff = (ref - result).abs().max().item()
        assert max_diff < 1e-5, f"RMSNorm+FiLM max diff: {max_diff}"

    def test_rmsnorm_no_film_correctness(self):
        """Metal RMSNorm (no FiLM) matches JIT version."""
        from star_ldm.models.modules.fused_blocks import (
            _jit_fused_rmsnorm, metal_rmsnorm,
        )

        B, L, D = 2, 8, 768
        x = torch.randn(B, L, D, device='mps')
        gamma = torch.randn(D, device='mps')
        dim_scale = math.sqrt(D)

        ref = _jit_fused_rmsnorm(x, gamma, dim_scale)
        result = metal_rmsnorm(x, gamma, dim_scale)

        if result is None:
            pytest.skip("Metal RMSNorm kernel not compiled")

        torch.mps.synchronize()
        max_diff = (ref - result).abs().max().item()
        assert max_diff < 1e-5, f"RMSNorm max diff: {max_diff}"


# ---- Tiny Attention Kernel ----

class TestMetalTinyAttention:
    """Compare Metal 8-token attention kernel against JIT-scripted version."""

    @pytest.fixture
    def tensors(self):
        B, H, S, D_head = 2, 8, 8, 96
        q = torch.randn(B, H, S, D_head, device='mps')
        k = torch.randn(B, H, S, D_head, device='mps')
        v = torch.randn(B, H, S, D_head, device='mps')
        q_gamma = torch.randn(D_head, device='mps')
        k_gamma = torch.randn(D_head, device='mps')
        dim_head_scale = math.sqrt(D_head)
        attn_scale = 1.0 / math.sqrt(D_head)
        return q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale

    def test_tiny_attention_correctness(self, tensors):
        """Metal tiny attention matches JIT version."""
        from star_ldm.models.modules.fused_blocks import (
            _jit_fused_qknorm_attention, metal_tiny_attention,
        )

        q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale = tensors

        ref = _jit_fused_qknorm_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)
        result = metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)

        if result is None:
            pytest.skip("Metal tiny attention kernel not compiled")

        torch.mps.synchronize()
        max_diff = (ref - result).abs().max().item()
        assert max_diff < 1e-4, f"Tiny attention max diff: {max_diff}"

    def test_attention_batch_sizes(self):
        """Test various batch sizes for tiny attention."""
        from star_ldm.models.modules.fused_blocks import (
            _jit_fused_qknorm_attention, metal_tiny_attention,
        )

        for B in [1, 2, 4]:
            H, S, D_head = 8, 8, 96
            q = torch.randn(B, H, S, D_head, device='mps')
            k = torch.randn(B, H, S, D_head, device='mps')
            v = torch.randn(B, H, S, D_head, device='mps')
            q_gamma = torch.randn(D_head, device='mps')
            k_gamma = torch.randn(D_head, device='mps')

            ref = _jit_fused_qknorm_attention(q, k, v, q_gamma, k_gamma, math.sqrt(D_head), 1.0/math.sqrt(D_head))
            result = metal_tiny_attention(q, k, v, q_gamma, k_gamma, math.sqrt(D_head), 1.0/math.sqrt(D_head))

            if result is None:
                pytest.skip("Metal tiny attention kernel not compiled")

            torch.mps.synchronize()
            max_diff = (ref - result).abs().max().item()
            assert max_diff < 1e-4, f"B={B}: max diff {max_diff}"


# ---- Speculative Verification Kernel ----

class TestMetalSpecVerify:
    """Compare Metal speculative verification against PyTorch fallback."""

    def test_spec_verify_all_accept(self):
        """When target probs >> draft probs, all tokens should be accepted."""
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 50257
        # Make target strongly prefer the draft tokens
        draft_logits = torch.randn(K, V, device='mps')
        target_logits = draft_logits * 2.0  # Target more confident

        # Sample tokens from draft
        draft_probs = torch.softmax(draft_logits, dim=-1)
        draft_tokens = torch.argmax(draft_probs, dim=-1)

        # Use very small random values to ensure acceptance
        rand_uniform = torch.zeros(K, device='mps')

        first_reject, adjusted = _verify_candidates_torch(
            draft_logits, target_logits, draft_tokens, rand_uniform
        )

        assert first_reject == K, f"Expected all accepted, got rejection at {first_reject}"

    def test_spec_verify_rejection(self):
        """When draft and target disagree, rejection should occur."""
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 1000  # Smaller vocab for test speed

        # Draft strongly prefers token 0, target strongly prefers token 1
        draft_logits = torch.zeros(K, V, device='mps')
        draft_logits[:, 0] = 10.0

        target_logits = torch.zeros(K, V, device='mps')
        target_logits[:, 1] = 10.0

        draft_tokens = torch.zeros(K, device='mps', dtype=torch.long)  # token 0

        # Use high random value to trigger rejection
        rand_uniform = torch.ones(K, device='mps') * 0.99

        first_reject, adjusted = _verify_candidates_torch(
            draft_logits, target_logits, draft_tokens, rand_uniform
        )

        assert first_reject == 0, f"Expected rejection at 0, got {first_reject}"
        # Adjusted probs should favor token 1
        assert adjusted[1].item() > adjusted[0].item()

    def test_metal_vs_torch_verify(self):
        """Metal verification matches PyTorch fallback."""
        try:
            from star_ldm.kernels import get_spec_verify_kernel
            metal_mod = get_spec_verify_kernel()
            if metal_mod is None:
                pytest.skip("Metal spec_verify kernel not compiled")
        except Exception:
            pytest.skip("Metal spec_verify kernel not available")

        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 50257
        draft_logits = torch.randn(K, V, device='mps')
        target_logits = torch.randn(K, V, device='mps')
        draft_probs = torch.softmax(draft_logits, dim=-1)
        draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(-1)
        rand_uniform = torch.rand(K, device='mps')

        # PyTorch reference
        ref_reject, ref_adjusted = _verify_candidates_torch(
            draft_logits, target_logits, draft_tokens, rand_uniform
        )

        # Metal
        metal_reject_t, metal_adjusted = metal_mod.speculative_verify(
            draft_logits.contiguous(), target_logits.contiguous(),
            draft_tokens.contiguous(), rand_uniform.contiguous(),
        )
        torch.mps.synchronize()
        metal_reject = metal_reject_t.item()

        assert ref_reject == metal_reject, f"Rejection index mismatch: ref={ref_reject}, metal={metal_reject}"

        if ref_reject < K:
            max_diff = (ref_adjusted - metal_adjusted).abs().max().item()
            assert max_diff < 1e-4, f"Adjusted probs max diff: {max_diff}"
