"""Canonical evaluation: full cross-product of step counts x prefix lengths x configs.

Both 20 and 50 steps, both baseline and optimized, 5 prefix lengths, 10 runs each.
Produces the definitive results table with error bars.

Usage:
    python -m scripts.eval_canonical --model_path checkpoints/star-ldm
"""
import argparse, json, time, torch
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    device = 'mps'
    step_counts = [20, 50]
    prefix_targets = [16, 64, 128, 256, 512]

    # Build long prompt
    base_text = "The quick brown fox jumps over the lazy dog and runs across the field towards the distant mountain range where the sun sets beautifully every evening creating magnificent colors in the sky. " * 20
    from star_ldm.interface import TransfusionGPTInterface
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks

    # Load BASELINE model (no optimizations)
    print("Loading baseline model...")
    intf_base = TransfusionGPTInterface(model_path=args.model_path, device=device)
    model_base = intf_base.model
    tok = model_base.tokenizer
    gk = {"do_sample": True, "max_new_tokens": 32, "pad_token_id": tok.eos_token_id,
          "top_p": 0.9, "repetition_penalty": 1.2}

    tokens_full = tok(base_text, return_tensors="pt").input_ids[0]
    max_avail = len(tokens_full)
    prefix_targets = [p for p in prefix_targets if p <= max_avail]
    print(f"Available tokens: {max_avail}, testing prefixes: {prefix_targets}")

    # Load OPTIMIZED model
    print("Loading optimized model...")
    intf_opt = TransfusionGPTInterface(model_path=args.model_path, device=device)
    model_opt = intf_opt.model
    swap_to_fused_blocks(model_opt.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model_opt.score_net_head.transformer)

    results = {}

    for steps in step_counts:
        for pfx in prefix_targets:
            ids = tokens_full[:pfx].unsqueeze(0).to(device)
            key = f"steps={steps}_pfx={pfx}"
            print(f"\n{'='*60}")
            print(f"  {steps} steps, prefix={pfx} tokens, {args.runs} runs each")
            print(f"{'='*60}")

            # ── BASELINE ──
            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    model_base.sample(ids, sampling_timesteps=steps, generate_kwargs=gk)

            base_times = []
            for r in range(args.runs):
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model_base.sample(ids, sampling_timesteps=steps, generate_kwargs=gk)
                torch.mps.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                base_times.append(ms)

            # ── OPTIMIZED ──
            for _ in range(2):
                with torch.no_grad():
                    model_opt.sample_with_kv_cache(ids, sampling_timesteps=steps, generate_kwargs=gk)

            opt_times = []
            for r in range(args.runs):
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model_opt.sample_with_kv_cache(ids, sampling_timesteps=steps, generate_kwargs=gk)
                torch.mps.synchronize()
                ms = (time.perf_counter() - t0) * 1000
                opt_times.append(ms)

            ba = np.array(base_times)
            oa = np.array(opt_times)
            speedup = ba.mean() / oa.mean()

            results[key] = {
                'steps': steps,
                'prefix_len': pfx,
                'baseline': {
                    'mean': float(ba.mean()),
                    'std': float(ba.std()),
                    'ci95': float(1.96 * ba.std() / np.sqrt(len(ba))),
                    'median': float(np.median(ba)),
                    'min': float(ba.min()),
                    'max': float(ba.max()),
                    'times': [float(t) for t in base_times],
                },
                'optimized': {
                    'mean': float(oa.mean()),
                    'std': float(oa.std()),
                    'ci95': float(1.96 * oa.std() / np.sqrt(len(oa))),
                    'median': float(np.median(oa)),
                    'min': float(oa.min()),
                    'max': float(oa.max()),
                    'times': [float(t) for t in opt_times],
                },
                'speedup': float(speedup),
            }

            print(f"  Baseline:  {ba.mean():.0f} +/- {ba.std():.0f} ms")
            print(f"  Optimized: {oa.mean():.0f} +/- {oa.std():.0f} ms")
            print(f"  Speedup:   {speedup:.2f}x")

    # ══ SUMMARY TABLES ══
    for steps in step_counts:
        print(f"\n{'='*70}")
        print(f"  CANONICAL RESULTS: {steps} DIFFUSION STEPS ({args.runs} runs)")
        print(f"{'='*70}")
        print(f"  {'Prefix':>6}  {'Baseline':>14}  {'Optimized':>14}  {'Speedup':>8}")
        print(f"  {'-'*48}")
        for pfx in prefix_targets:
            key = f"steps={steps}_pfx={pfx}"
            if key not in results:
                continue
            r = results[key]
            b = r['baseline']
            o = r['optimized']
            print(f"  {pfx:>6}  {b['mean']:>7.0f} +/-{b['ci95']:>4.0f}  {o['mean']:>7.0f} +/-{o['ci95']:>4.0f}  {r['speedup']:>7.2f}x")

    # Save
    out = {
        'runs_per_config': args.runs,
        'step_counts': step_counts,
        'prefix_lengths': prefix_targets,
        'device': device,
        'results': results,
    }
    out_path = 'results_canonical.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
