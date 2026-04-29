"""Batch size sweep for throughput scaling analysis.

Measures end-to-end latency and throughput as batch size increases,
with the best optimization config (fused blocks + KV-cache).

Usage:
    python -m scripts.batch_sweep --model_path checkpoints/star-ldm
"""

import argparse
import time
import json
import torch
from omegaconf import OmegaConf


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--prompt', type=str, default='The meaning of life is')
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    from star_ldm.interface import TransfusionGPTInterface
    interface = TransfusionGPTInterface(model_path=args.model_path, device=str(device))
    model = interface.model

    # Enable fused blocks
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks
    swap_to_fused_blocks(model.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model.score_net_head.transformer)
    print("Fused blocks activated.")

    tokenizer = model.tokenizer
    results = {}

    for B in batch_sizes:
        print(f"\n{'='*60}")
        print(f"  Batch size = {B}")
        print(f"{'='*60}")

        # Create batched input
        input_ids = tokenizer(
            [args.prompt] * B, return_tensors='pt', padding=True
        ).input_ids.to(device)

        # Warmup
        for _ in range(args.warmup):
            with torch.no_grad():
                try:
                    model.sample_with_kv_cache(
                        input_ids,
                        sampling_timesteps=args.sampling_timesteps,
                        generate_kwargs={
                            "do_sample": True, "max_new_tokens": 32,
                            "pad_token_id": tokenizer.eos_token_id,
                            "top_p": 0.9, "repetition_penalty": 1.2,
                        },
                    )
                except Exception as e:
                    print(f"  Warmup error: {e}")
                    break

        # Timed runs
        times = []
        for i in range(args.num_runs):
            with torch.no_grad():
                sync(device)
                t0 = time.perf_counter()
                try:
                    model.sample_with_kv_cache(
                        input_ids,
                        sampling_timesteps=args.sampling_timesteps,
                        generate_kwargs={
                            "do_sample": True, "max_new_tokens": 32,
                            "pad_token_id": tokenizer.eos_token_id,
                            "top_p": 0.9, "repetition_penalty": 1.2,
                        },
                    )
                except Exception as e:
                    print(f"  Run {i+1}: ERROR - {e}")
                    continue
                sync(device)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                times.append(elapsed_ms)
                print(f"  Run {i+1}: {elapsed_ms:.1f} ms  ({elapsed_ms/B:.1f} ms/sample)")

        if times:
            import numpy as np
            arr = np.array(times)
            results[B] = {
                'batch_size': B,
                'mean_ms': float(np.mean(arr)),
                'std_ms': float(np.std(arr)),
                'per_sample_ms': float(np.mean(arr)) / B,
                'throughput_samples_per_sec': B / (float(np.mean(arr)) / 1000),
            }
            print(f"  => Mean: {results[B]['mean_ms']:.1f} ms, "
                  f"Per-sample: {results[B]['per_sample_ms']:.1f} ms, "
                  f"Throughput: {results[B]['throughput_samples_per_sec']:.2f} samples/s")

    # Summary
    print(f"\n{'='*70}")
    print(f"  BATCH SIZE SCALING ({args.sampling_timesteps} steps, fused+KV-cache)")
    print(f"{'='*70}")
    print(f"  {'B':>4} {'Total ms':>10} {'Per-sample':>12} {'Throughput':>14} {'Scaling':>10}")
    print(f"  {'-'*54}")

    base_throughput = None
    for B in batch_sizes:
        if B not in results:
            continue
        r = results[B]
        if base_throughput is None:
            base_throughput = r['throughput_samples_per_sec']
        scaling = r['throughput_samples_per_sec'] / base_throughput
        print(f"  {B:>4} {r['mean_ms']:>10.1f} {r['per_sample_ms']:>10.1f} ms "
              f"{r['throughput_samples_per_sec']:>12.2f}/s {scaling:>9.2f}x")

    # Save
    out = {
        'sampling_timesteps': args.sampling_timesteps,
        'device': str(device),
        'config': 'fused + KV-cache',
        'results': results,
    }
    with open('results_batch_sweep.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to results_batch_sweep.json")


if __name__ == '__main__':
    main()
