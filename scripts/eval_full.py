"""Full evaluation on 100 diverse prompts across prefix length buckets.

Compares: baseline (HuggingFace) vs optimized (fast GPT-2 + fused + KV-cache)
at multiple step counts.

Usage:
    python -m scripts.eval_full --model_path checkpoints/star-ldm
"""
import argparse, json, time, torch
import numpy as np
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_prompts', type=int, default=100)
    parser.add_argument('--steps', type=str, default='50,20')
    args = parser.parse_args()

    device = 'mps'
    step_counts = [int(s) for s in args.steps.split(',')]

    from star_ldm.interface import TransfusionGPTInterface
    from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks

    print("Loading model...")
    interface = TransfusionGPTInterface(model_path=args.model_path, device=device)
    model = interface.model
    swap_to_fused_blocks(model.soft_prompt_generator.transformer)
    swap_to_fused_blocks(model.score_net_head.transformer)
    tok = model.tokenizer

    with open('data/eval_prompts.json') as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts")

    gk = {"do_sample": True, "max_new_tokens": 64, "pad_token_id": tok.eos_token_id,
          "top_p": 0.9, "repetition_penalty": 1.2}

    # Warmup
    print("Warming up...")
    ids = tok("warmup", return_tensors='pt').input_ids.to(device)
    for _ in range(3):
        with torch.no_grad():
            model.sample_with_kv_cache(ids, sampling_timesteps=20, generate_kwargs=gk)

    results = {}
    for steps in step_counts:
        print(f"\n{'='*60}")
        print(f"  {steps} diffusion steps — {len(prompts)} prompts")
        print(f"{'='*60}")

        bucket_times = defaultdict(list)
        all_times = []

        for i, p in enumerate(prompts):
            text = p['text']
            bucket = p['bucket']
            input_ids = tok(text, return_tensors='pt').input_ids.to(device)
            prefix_len = input_ids.shape[1]

            torch.mps.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                emb, gen = model.sample_with_kv_cache(
                    input_ids, sampling_timesteps=steps, generate_kwargs=gk)
            torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            all_times.append(elapsed_ms)
            bucket_times[bucket].append(elapsed_ms)

            if i < 5 or (i + 1) % 25 == 0:
                print(f"  [{i+1:>3}/{len(prompts)}] pfx={prefix_len:>3}tok "
                      f"{elapsed_ms:>7.0f}ms  {bucket:<20s} {gen[0][:40]}...")

        arr = np.array(all_times)
        results[steps] = {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            'p5': float(np.percentile(arr, 5)),
            'p95': float(np.percentile(arr, 95)),
            'per_bucket': {},
        }
        for bucket, times in sorted(bucket_times.items()):
            ba = np.array(times)
            results[steps]['per_bucket'][bucket] = {
                'mean': float(np.mean(ba)),
                'median': float(np.median(ba)),
                'std': float(np.std(ba)),
                'count': len(times),
            }

        print(f"\n  RESULTS ({steps} steps):")
        print(f"  Overall: mean={results[steps]['mean']:.0f}ms  "
              f"median={results[steps]['median']:.0f}ms  "
              f"std={results[steps]['std']:.0f}ms  "
              f"p5={results[steps]['p5']:.0f}ms  p95={results[steps]['p95']:.0f}ms")
        for bucket in sorted(results[steps]['per_bucket'].keys()):
            b = results[steps]['per_bucket'][bucket]
            print(f"  {bucket:<20s}: mean={b['mean']:.0f}ms  median={b['median']:.0f}ms  (n={b['count']})")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for steps in step_counts:
        r = results[steps]
        speedup = 2346 / r['mean']  # vs unoptimized baseline from results_c4_50.json
        print(f"  {steps} steps: {r['mean']:.0f}ms mean  ({speedup:.2f}x vs unoptimized baseline 2346ms)")

    out_path = f'results_eval_{args.num_prompts}prompts.json'
    with open(out_path, 'w') as f:
        json.dump({'step_counts': step_counts, 'num_prompts': len(prompts), 'results': results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
