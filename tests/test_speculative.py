"""Tests for speculative decoding module.

Tests correctness of the custom speculative decoding loop including
acceptance criterion, KV cache truncation, and sampling utilities.
"""

import pytest
import torch
import torch.nn.functional as F


class TestApplySampling:
    """Test the sampling utility function."""

    def test_temperature_scaling(self):
        from star_ldm.decoding.speculative import _apply_sampling

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        # High temperature should flatten distribution
        high_temp = _apply_sampling(logits, temperature=10.0, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])
        probs_high = F.softmax(high_temp, dim=-1)

        low_temp = _apply_sampling(logits, temperature=0.1, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])
        probs_low = F.softmax(low_temp, dim=-1)

        # High temp → more uniform; low temp → more peaked
        assert probs_high.max() < probs_low.max()

    def test_repetition_penalty(self):
        from star_ldm.decoding.speculative import _apply_sampling

        logits = torch.tensor([[5.0, 1.0, 1.0]])
        # Without penalty
        no_pen = _apply_sampling(logits, temperature=1.0, top_p=1.0, repetition_penalty=1.0, prev_tokens=[])
        # With penalty on token 0
        with_pen = _apply_sampling(logits.clone(), temperature=1.0, top_p=1.0, repetition_penalty=2.0, prev_tokens=[0])

        # Penalized logit for token 0 should be smaller
        assert with_pen[0, 0] < no_pen[0, 0]

    def test_top_p_filtering(self):
        from star_ldm.decoding.speculative import _apply_sampling

        # One very dominant token
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        result = _apply_sampling(logits, temperature=1.0, top_p=0.5, repetition_penalty=1.0, prev_tokens=[])

        probs = F.softmax(result, dim=-1)
        # Token 0 should have almost all probability
        assert probs[0, 0].item() > 0.99


class TestTruncateKVCache:
    """Test KV cache truncation."""

    def test_truncate(self):
        from star_ldm.decoding.speculative import _truncate_kv_cache

        # Simulate 2-layer KV cache, batch=1, heads=2, total_seq=10, dim=4
        past = tuple(
            (torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
            for _ in range(2)
        )

        # Added 4 tokens, keep 2
        truncated = _truncate_kv_cache(past, current_len=4, keep_len=2)

        for key, value in truncated:
            assert key.shape[2] == 8  # 10 - 2 removed
            assert value.shape[2] == 8

    def test_truncate_noop(self):
        from star_ldm.decoding.speculative import _truncate_kv_cache

        past = tuple(
            (torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4))
            for _ in range(2)
        )

        # Keep all
        result = _truncate_kv_cache(past, current_len=4, keep_len=4)

        # Should be the same object (no-op)
        assert result is past


class TestVerifyCandidates:
    """Test the verification logic."""

    def test_certain_accept(self):
        """When target == draft, all tokens should be accepted."""
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 100
        logits = torch.randn(K, V)
        tokens = torch.argmax(logits, dim=-1)
        rand_uniform = torch.zeros(K)  # Always accept

        reject_idx, _ = _verify_candidates_torch(logits, logits, tokens, rand_uniform)
        assert reject_idx == K

    def test_certain_reject(self):
        """When distributions disagree maximally, first token should reject."""
        from star_ldm.decoding.speculative import _verify_candidates_torch

        K, V = 4, 100
        draft_logits = torch.zeros(K, V)
        draft_logits[:, 0] = 100.0  # Draft loves token 0

        target_logits = torch.zeros(K, V)
        target_logits[:, 1] = 100.0  # Target loves token 1

        tokens = torch.zeros(K, dtype=torch.long)  # Draft sampled token 0
        rand_uniform = torch.ones(K) * 0.5  # Moderate threshold

        reject_idx, adjusted = _verify_candidates_torch(
            draft_logits, target_logits, tokens, rand_uniform
        )
        assert reject_idx == 0
        # Adjusted should strongly favor token 1
        assert adjusted[1].item() > 0.9
