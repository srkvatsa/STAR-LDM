"""Microbenchmarks for Metal compute shaders.

Measures kernel execution time for each Metal shader and compares
against the JIT-scripted fallback. Reports speedup and estimates
roofline characteristics.

Usage:
    python scripts/benchmark_metal.py [--warmup 10] [--iterations 100]
"""

import argparse
import time
import math
import torch

# Check MPS availability
if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    print("MPS not available. This benchmark requires Apple Silicon.")
    exit(1)


def sync():
    """Synchronize MPS device."""
    torch.mps.synchronize()


def benchmark_fn(fn, args, warmup=10, iterations=100, label=""):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    sync()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    sync()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iterations * 1000
    print(f"  {label}: {avg_ms:.4f} ms/call ({iterations} iters)")
    return avg_ms


def benchmark_ddpm_step(B=1, D=768, warmup=10, iterations=100):
    """Benchmark DDPM step kernel."""
    from star_ldm.diffusion.fused_ops import _jit_fused_ddpm_step, metal_ddpm_step

    print(f"\n=== DDPM Step Kernel (B={B}, D={D}) ===")

    z_t = torch.randn(B, D, device='mps')
    eps = torch.randn(B, D, device='mps')
    noise = torch.randn(B, D, device='mps')
    alpha2 = torch.rand(B, 1, device='mps') * 0.8 + 0.1
    alpha2_next = alpha2 * 0.5
    var_lambda = 0.2

    jit_time = benchmark_fn(
        _jit_fused_ddpm_step,
        (z_t, eps, noise, alpha2, alpha2_next, var_lambda),
        warmup=warmup, iterations=iterations, label="JIT"
    )

    metal_fn = lambda *a: metal_ddpm_step(*a)
    result = metal_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, var_lambda)
    if result is not None:
        metal_time = benchmark_fn(
            metal_fn,
            (z_t, eps, noise, alpha2, alpha2_next, var_lambda),
            warmup=warmup, iterations=iterations, label="Metal"
        )
        print(f"  Speedup: {jit_time/metal_time:.2f}x")
    else:
        print("  Metal kernel not available, skipping.")

    # Roofline analysis
    bytes_accessed = B * D * 4 * 4  # 3 inputs + 1 output, float32
    flops = B * D * 20  # ~20 ops per element
    print(f"  Bytes: {bytes_accessed/1024:.1f} KB, FLOPs: {flops/1000:.1f} KFLOP")
    print(f"  Arithmetic intensity: {flops/bytes_accessed:.2f} FLOP/byte (memory-bound)")


def benchmark_rmsnorm_film(B=2, L=8, D=768, warmup=10, iterations=100):
    """Benchmark RMSNorm+FiLM kernel."""
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_rmsnorm_film, metal_rmsnorm_film,
    )

    print(f"\n=== RMSNorm+FiLM Kernel (B={B}, L={L}, D={D}) ===")

    x = torch.randn(B, L, D, device='mps')
    gamma = torch.randn(D, device='mps')
    dim_scale = math.sqrt(D)
    film_scale = torch.randn(B, 1, D, device='mps')
    film_shift = torch.randn(B, 1, D, device='mps')

    jit_time = benchmark_fn(
        _jit_fused_rmsnorm_film,
        (x, gamma, dim_scale, film_scale, film_shift),
        warmup=warmup, iterations=iterations, label="JIT"
    )

    result = metal_rmsnorm_film(x, gamma, dim_scale, film_scale, film_shift)
    if result is not None:
        metal_time = benchmark_fn(
            metal_rmsnorm_film,
            (x, gamma, dim_scale, film_scale, film_shift),
            warmup=warmup, iterations=iterations, label="Metal"
        )
        print(f"  Speedup: {jit_time/metal_time:.2f}x")
    else:
        print("  Metal kernel not available, skipping.")


def benchmark_tiny_attention(B=2, H=8, S=8, D_head=96, warmup=10, iterations=100):
    """Benchmark 8-token attention kernel."""
    from star_ldm.models.modules.fused_blocks import (
        _jit_fused_qknorm_attention, metal_tiny_attention,
    )

    print(f"\n=== Tiny Attention Kernel (B={B}, H={H}, S={S}, D_head={D_head}) ===")

    q = torch.randn(B, H, S, D_head, device='mps')
    k = torch.randn(B, H, S, D_head, device='mps')
    v = torch.randn(B, H, S, D_head, device='mps')
    q_gamma = torch.randn(D_head, device='mps')
    k_gamma = torch.randn(D_head, device='mps')
    dim_head_scale = math.sqrt(D_head)
    attn_scale = 1.0 / math.sqrt(D_head)

    jit_time = benchmark_fn(
        _jit_fused_qknorm_attention,
        (q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale),
        warmup=warmup, iterations=iterations, label="JIT"
    )

    result = metal_tiny_attention(q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale)
    if result is not None:
        metal_time = benchmark_fn(
            metal_tiny_attention,
            (q, k, v, q_gamma, k_gamma, dim_head_scale, attn_scale),
            warmup=warmup, iterations=iterations, label="Metal"
        )
        print(f"  Speedup: {jit_time/metal_time:.2f}x")
    else:
        print("  Metal kernel not available, skipping.")

    # Roofline
    bytes_qkv = B * H * S * D_head * 4 * 3  # Q, K, V reads
    bytes_out = B * H * S * D_head * 4  # output write
    flops_qk = B * H * S * S * D_head * 2  # QK^T
    flops_av = B * H * S * D_head * S * 2   # Attn * V
    total_flops = flops_qk + flops_av
    total_bytes = bytes_qkv + bytes_out
    print(f"  Bytes: {total_bytes/1024:.1f} KB, FLOPs: {total_flops/1000:.1f} KFLOP")
    print(f"  Arithmetic intensity: {total_flops/total_bytes:.2f} FLOP/byte")


def benchmark_spec_verify(K=4, V=50257, warmup=10, iterations=100):
    """Benchmark speculative verification kernel."""
    from star_ldm.decoding.speculative import _verify_candidates_torch

    print(f"\n=== Speculative Verification (K={K}, V={V}) ===")

    draft_logits = torch.randn(K, V, device='mps')
    target_logits = torch.randn(K, V, device='mps')
    draft_probs = torch.softmax(draft_logits, dim=-1)
    draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(-1)
    rand_uniform = torch.rand(K, device='mps')

    torch_time = benchmark_fn(
        _verify_candidates_torch,
        (draft_logits, target_logits, draft_tokens, rand_uniform),
        warmup=warmup, iterations=iterations, label="PyTorch"
    )

    try:
        from star_ldm.kernels import get_spec_verify_kernel
        metal_mod = get_spec_verify_kernel()
        if metal_mod is not None:
            metal_fn = lambda dl, tl, dt, ru: metal_mod.speculative_verify(
                dl.contiguous(), tl.contiguous(), dt.contiguous(), ru.contiguous()
            )
            metal_time = benchmark_fn(
                metal_fn,
                (draft_logits, target_logits, draft_tokens, rand_uniform),
                warmup=warmup, iterations=iterations, label="Metal"
            )
            print(f"  Speedup: {torch_time/metal_time:.2f}x")
        else:
            print("  Metal kernel not available, skipping.")
    except Exception as e:
        print(f"  Metal kernel load error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Metal kernel microbenchmarks")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("STAR-LDM Metal Kernel Microbenchmarks")
    print(f"Device: MPS (Apple Silicon)")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print("=" * 60)

    # Use actual STAR-LDM model dimensions:
    # Micro-transformer: dim=1024, dim_head=64, 16 heads, seq_len=8
    # Sentence embedding: D=768
    benchmark_ddpm_step(B=1, D=768, warmup=args.warmup, iterations=args.iterations)
    benchmark_ddpm_step(B=4, D=768, warmup=args.warmup, iterations=args.iterations)
    benchmark_rmsnorm_film(B=1, L=8, D=1024, warmup=args.warmup, iterations=args.iterations)
    benchmark_tiny_attention(B=1, H=16, S=8, D_head=64, warmup=args.warmup, iterations=args.iterations)
    benchmark_spec_verify(warmup=args.warmup, iterations=args.iterations)

    print("\n" + "=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
