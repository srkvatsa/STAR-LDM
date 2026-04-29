"""Validate that sample_with_kv_cache produces identical results to sample.

Usage:
    python -m scripts.validate_kv_cache --model_path checkpoints/star-ldm
    python -m scripts.validate_kv_cache --model_path checkpoints/star-ldm --sampling_timesteps 10
"""

import argparse
import time
import torch
from star_ldm.interface import TransfusionGPTInterface


def validate(model_path: str, sampling_timesteps: int = 50, device: str = 'cuda'):
    print(f"Loading model from {model_path}...")
    interface = TransfusionGPTInterface(model_path, device=device)
    model = interface.model
    tokenizer = interface.tokenizer
    actual_device = interface.device

    prompts = [
        "The quick brown fox",
        "Scientists have discovered that",
        "In a surprising turn of events,",
    ]

    kwargs = dict(
        sampling_timesteps=sampling_timesteps,
        sampler='ddpm',
        var_lambda=0.2,
        sigma2=0.05,
        cosine_scale=3.0,
    )

    print(f"\nValidating with {sampling_timesteps} diffusion steps on {actual_device}")
    print("=" * 70)

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(actual_device)
        print(f"\nPrompt: {prompt!r}  (prefix_len={input_ids.shape[1]})")

        # Run original sample() with fixed seed
        torch.manual_seed(42)
        t0 = time.perf_counter()
        x_orig, gen_orig = model.sample(input_ids, **kwargs)
        if actual_device.type == 'mps':
            torch.mps.synchronize()
        elif actual_device.type == 'cuda':
            torch.cuda.synchronize()
        t_orig = time.perf_counter() - t0

        # Run sample_with_kv_cache() with same seed
        torch.manual_seed(42)
        t0 = time.perf_counter()
        x_cached, gen_cached = model.sample_with_kv_cache(input_ids, **kwargs)
        if actual_device.type == 'mps':
            torch.mps.synchronize()
        elif actual_device.type == 'cuda':
            torch.cuda.synchronize()
        t_cached = time.perf_counter() - t0

        # Compare embeddings
        emb_diff = (x_orig - x_cached).abs().max().item()
        emb_l2 = (x_orig - x_cached).norm().item()

        # Compare generations
        gen_match = gen_orig[0] == gen_cached[0]

        print(f"  Original:  {t_orig*1000:.0f}ms  | gen: {gen_orig[0][:80]}...")
        print(f"  KV-cache:  {t_cached*1000:.0f}ms  | gen: {gen_cached[0][:80]}...")
        print(f"  Speedup:   {t_orig/t_cached:.2f}x")
        print(f"  Emb max diff: {emb_diff:.6e}  |  L2: {emb_l2:.6e}")
        print(f"  Generation match: {gen_match}")

        if emb_diff > 1e-3:
            print("  WARNING: Embedding difference is large!")
        if not gen_match:
            print("  NOTE: Generation mismatch (may be due to sampling randomness in generate)")

    print("\n" + "=" * 70)
    print("Validation complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    validate(args.model_path, args.sampling_timesteps, args.device)
