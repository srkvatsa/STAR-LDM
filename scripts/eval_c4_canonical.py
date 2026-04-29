"""Canonical evaluation on real C4 validation prompts.

Uses real C4 text at each prefix length bucket. Matches the original
STAR-LDM paper's evaluation data source.

Usage:
    python -m scripts.eval_c4_canonical --model_path checkpoints/star-ldm
"""
import argparse, json, time, torch
import numpy as np
from datasets import load_dataset


def get_c4_prompts(tokenizer, prefix_len, n_prompts=20, seed=42):
    """Get n_prompts real C4 texts tokenized to exactly prefix_len tokens."""
    ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=50000)

    prompts = []
    for sample in ds:
        ids = tokenizer(sample['text'], return_tensors='pt', truncation=False).input_ids[0]
        # Need at least prefix_len + 8 tokens (soft prompts) + some headroom
        if len(ids) >= prefix_len + 16:
            prompts.append(ids[:prefix_len])
            if len(prompts) >= n_prompts:
                break
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--prompts_per_bucket', type=int, default=20)
    args = parser.parse_args()

    device = 'mps'
    step_counts = [20, 50]
    prefix_lengths = [16, 64, 128, 256, 512, 768, 1000]

    from star_ldm.interface import TransfusionGPTInterface
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks

    print("Loading models...")
    intf_base = TransfusionGPTInterface(model_path=args.model_path, device=device)
    model_base = intf_base.model
    intf_opt = TransfusionGPTInterface(model_path=args.model_path, device=device)
    model_opt = intf_opt.model
    swap_to_fused_blocks(model_opt.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model_opt.score_net_head.transformer)
    tok = model_base.tokenizer

    # Gather real C4 prompts for each prefix length
    print("Fetching C4 validation prompts...")
    prompts_by_len = {}
    for pfx in prefix_lengths:
        prompts_by_len[pfx] = get_c4_prompts(tok, pfx, n_prompts=args.prompts_per_bucket)
        print(f"  prefix={pfx}: got {len(prompts_by_len[pfx])} prompts")

    results = {}
    for steps in step_counts:
        print(f"\n{'='*70}")
        print(f"  {steps} DIFFUSION STEPS, {args.runs} runs per prompt")
        print(f"{'='*70}")

        for pfx in prefix_lengths:
            c4_prompts = prompts_by_len[pfx]
            if len(c4_prompts) == 0:
                print(f"  prefix={pfx}: no prompts available, skipping")
                continue

            max_new = min(32, 1024 - pfx - 8)
            if max_new < 4:
                print(f"  prefix={pfx}: no room for generation, skipping")
                continue

            gk = {"do_sample": True, "max_new_tokens": max_new,
                  "pad_token_id": tok.eos_token_id, "top_p": 0.9, "repetition_penalty": 1.2}

            base_all, opt_all = [], []

            for p_idx, prompt_ids in enumerate(c4_prompts):
                ids = prompt_ids.unsqueeze(0).to(device)

                # Warmup on first prompt only
                if p_idx == 0:
                    for _ in range(2):
                        with torch.no_grad():
                            model_base.sample(ids, sampling_timesteps=steps, generate_kwargs=gk)
                            model_opt.sample_with_kv_cache(ids, sampling_timesteps=steps, generate_kwargs=gk)

                for _ in range(args.runs):
                    torch.mps.synchronize(); t0 = time.perf_counter()
                    with torch.no_grad():
                        model_base.sample(ids, sampling_timesteps=steps, generate_kwargs=gk)
                    torch.mps.synchronize()
                    base_all.append((time.perf_counter() - t0) * 1000)

                    torch.mps.synchronize(); t0 = time.perf_counter()
                    with torch.no_grad():
                        model_opt.sample_with_kv_cache(ids, sampling_timesteps=steps, generate_kwargs=gk)
                    torch.mps.synchronize()
                    opt_all.append((time.perf_counter() - t0) * 1000)

                if (p_idx + 1) % 5 == 0:
                    print(f"    prefix={pfx}, prompt {p_idx+1}/{len(c4_prompts)}")

            ba = np.array(base_all)
            oa = np.array(opt_all)
            n = len(ba)
            sp = ba.mean() / oa.mean()

            key = f"steps={steps}_pfx={pfx}"
            results[key] = {
                'steps': steps, 'prefix_len': pfx,
                'n_prompts': len(c4_prompts), 'runs_per_prompt': args.runs,
                'total_measurements': n,
                'baseline': {
                    'mean': float(ba.mean()), 'std': float(ba.std()),
                    'ci95': float(1.96 * ba.std() / np.sqrt(n)),
                    'median': float(np.median(ba)),
                    'p5': float(np.percentile(ba, 5)),
                    'p95': float(np.percentile(ba, 95)),
                },
                'optimized': {
                    'mean': float(oa.mean()), 'std': float(oa.std()),
                    'ci95': float(1.96 * oa.std() / np.sqrt(n)),
                    'median': float(np.median(oa)),
                    'p5': float(np.percentile(oa, 5)),
                    'p95': float(np.percentile(oa, 95)),
                },
                'speedup': float(sp),
            }
            print(f"  pfx={pfx:>4}: base={ba.mean():>7.0f}+/-{results[key]['baseline']['ci95']:>3.0f}  "
                  f"opt={oa.mean():>7.0f}+/-{results[key]['optimized']['ci95']:>3.0f}  "
                  f"speedup={sp:.2f}x  (n={n})")

    # Summary tables
    for steps in step_counts:
        print(f"\n{'='*70}")
        print(f"  C4 CANONICAL RESULTS: {steps} STEPS")
        print(f"  ({args.prompts_per_bucket} real C4 prompts x {args.runs} runs per bucket)")
        print(f"{'='*70}")
        print(f"  {'Prefix':>6}  {'n':>5}  {'Baseline':>16}  {'Optimized':>16}  {'Speedup':>8}")
        print(f"  {'-'*58}")
        for pfx in prefix_lengths:
            key = f"steps={steps}_pfx={pfx}"
            if key not in results:
                continue
            r = results[key]
            b, o = r['baseline'], r['optimized']
            print(f"  {pfx:>6}  {r['total_measurements']:>5}  "
                  f"{b['mean']:>7.0f} +/-{b['ci95']:>4.0f} ms  "
                  f"{o['mean']:>7.0f} +/-{o['ci95']:>4.0f} ms  {r['speedup']:>7.2f}x")

    with open('results_c4_canonical.json', 'w') as f:
        json.dump({
            'dataset': 'allenai/c4 validation',
            'prompts_per_bucket': args.prompts_per_bucket,
            'runs_per_prompt': args.runs,
            'step_counts': step_counts,
            'prefix_lengths': prefix_lengths,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved to results_c4_canonical.json")


if __name__ == '__main__':
    main()
