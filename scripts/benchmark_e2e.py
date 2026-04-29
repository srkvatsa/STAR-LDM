"""
End-to-end benchmark comparing all optimization configurations.

Configs:
  1. TRUE BASELINE: unfused ops, no KV-cache, vanilla generate
  2. + Fused blocks (Metal kernels where available)
  3. + KV-cache reuse
  4. + Async CPU/GPU overlap
  5. + Speculative decoding (KV-cache + spec decode for AR step)

Usage:
  python -m scripts.benchmark_e2e --dummy --num_runs 5
  python -m scripts.benchmark_e2e --model_path checkpoints/star-ldm --num_runs 10
"""

import argparse
import time
import torch
from contextlib import contextmanager
from omegaconf import OmegaConf


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


@contextmanager
def timed(device):
    sync_device(device)
    t0 = time.perf_counter()
    result = {}
    yield result
    sync_device(device)
    result['ms'] = (time.perf_counter() - t0) * 1000


def create_dummy_model(device):
    cfg = OmegaConf.create({
        'dataset_name': 'fineweb_100b',
        'train': {'freeze_gpt': True, 'lm_name': 'gpt2-large', 'global_norm': True},
        'sampling': {'noise_schedule_name': 'cosine', 'noise_schedule_scale': 1.0},
        'diffusion_loss': {
            'weighting_name': 'sigmoid',
            'weighting_kwargs': {'gamma_shift': 0.0},
            'train_schedule': 'cosine',
            'cosine_shift': 0.0,
        },
        'prompt_generator': {'dim': 1024, 'dim_head': 64, 'depth': 6, 'prompt_length': 8, 'dropout': 0.0},
        'scorenet_head': {'dim': 1024, 'depth': 6, 'dropout': 0.0, 'output_dim_mult': 4},
    })
    from star_ldm.models.transfusion import TransfusionGPT
    model = TransfusionGPT(
        dataset_name='fineweb_100b',
        transfusion_cfg=cfg,
        gpt2_model_name='gpt2-large',
        gamma_min=-15, gamma_max=15,
        clf_guidance_dropout=0.1,
        scale_by_std=True,
        global_norm=True,
    )
    model = model.to(device).eval()
    return model


def run_baseline_unfused(model, input_ids, sampling_timesteps=50):
    """TRUE BASELINE: unfused ops, no KV-cache, standard generate."""
    return model.sample(
        input_ids,
        sampling_timesteps=sampling_timesteps,
        generate_kwargs={"do_sample": True, "max_new_tokens": 32,
                         "pad_token_id": model.tokenizer.eos_token_id,
                         "top_p": 0.9, "repetition_penalty": 1.2},
    )


def run_kv_cache(model, input_ids, sampling_timesteps=50):
    """KV-cache reuse (with fused ops)."""
    return model.sample_with_kv_cache(
        input_ids,
        sampling_timesteps=sampling_timesteps,
        generate_kwargs={"do_sample": True, "max_new_tokens": 32,
                         "pad_token_id": model.tokenizer.eos_token_id,
                         "top_p": 0.9, "repetition_penalty": 1.2},
    )


def run_async(model, input_ids, sampling_timesteps=50):
    """KV-cache + async CPU/GPU overlap."""
    return model.sample_with_async(
        input_ids,
        sampling_timesteps=sampling_timesteps,
        generate_kwargs={"do_sample": True, "max_new_tokens": 32,
                         "pad_token_id": model.tokenizer.eos_token_id,
                         "top_p": 0.9, "repetition_penalty": 1.2},
    )


def run_speculative(model, input_ids, target_model, sampling_timesteps=50, K=4):
    """KV-cache + speculative decoding for AR step (legacy: HF target model)."""
    return model.sample_with_speculative(
        input_ids,
        target_model=target_model,
        speculative_k=K,
        sampling_timesteps=sampling_timesteps,
        generate_kwargs={"do_sample": True, "max_new_tokens": 32,
                         "pad_token_id": model.tokenizer.eos_token_id,
                         "top_p": 0.9, "repetition_penalty": 1.2},
    )


def run_draft_speculative(model, input_ids, draft_model, sampling_timesteps=50, K=4):
    """KV-cache + distilled draft speculative decoding for AR step."""
    return model.sample_with_draft_speculative(
        input_ids,
        draft_model=draft_model,
        speculative_k=K,
        sampling_timesteps=sampling_timesteps,
        generate_kwargs={"do_sample": True, "max_new_tokens": 32,
                         "pad_token_id": model.tokenizer.eos_token_id,
                         "top_p": 0.9, "repetition_penalty": 1.2},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--prompt', type=str, default='The meaning of life is')
    parser.add_argument('--skip_spec', action='store_true', help='Skip speculative decoding')
    parser.add_argument('--draft_path', type=str, default=None,
                        help='Path to distilled draft model checkpoint')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    if args.dummy:
        model = create_dummy_model(device)
    else:
        from star_ldm.interface import TransfusionGPTInterface
        interface = TransfusionGPTInterface(model_path=args.model_path, device=str(device))
        model = interface.model

    tokenizer = model.tokenizer
    input_ids = tokenizer(args.prompt, return_tensors='pt').input_ids.to(device)
    print(f"Prompt: '{args.prompt}' ({input_ids.shape[1]} tokens)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load speculative decoding models
    target_model = None
    draft_model = None
    if not args.skip_spec:
        # Prefer distilled draft model if path provided
        if args.draft_path:
            print(f"Loading distilled draft model from {args.draft_path}...")
            from star_ldm.decoding.draft_model import load_draft_model
            try:
                draft_model = load_draft_model(args.draft_path, device=str(device))
                print(f"Draft model loaded: {draft_model.num_parameters:,} params")
            except Exception as e:
                print(f"Failed to load draft model: {e}")
                draft_model = None
        else:
            # Legacy: load GPT-2 XL as target
            print("Loading target model (gpt2-xl) for speculative decoding...")
            from transformers import AutoModelForCausalLM
            try:
                target_model = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device).eval()
                print(f"Target model loaded: {sum(p.numel() for p in target_model.parameters()):,} params")
            except Exception as e:
                print(f"Failed to load target model: {e}")
                target_model = None

    # ---------- Phase 1: unfused configs ----------
    configs = [
        ("1. BASELINE (unfused, no KV)", lambda: run_baseline_unfused(model, input_ids, args.sampling_timesteps)),
        ("2. + KV-cache only", lambda: run_kv_cache(model, input_ids, args.sampling_timesteps)),
    ]

    # ---------- Phase 2: activate fused blocks, then test fused configs ----------
    # swap_to_fused_blocks is destructive (in-place monkey-patch), so we run
    # all unfused configs first, then enable fusion for the remaining configs.
    FUSED_MARKER = "<<ACTIVATE_FUSED_BLOCKS>>"
    configs.append((FUSED_MARKER, None))

    configs.append(("3. + Fused micro-transformer ops", lambda: run_baseline_unfused(model, input_ids, args.sampling_timesteps)))
    configs.append(("4. + Fused + KV-cache", lambda: run_kv_cache(model, input_ids, args.sampling_timesteps)))
    configs.append(("5. + Fused + KV-cache + Async", lambda: run_async(model, input_ids, args.sampling_timesteps)))

    if draft_model is not None:
        configs.append(
            ("6. + Draft spec decode", lambda: run_draft_speculative(model, input_ids, draft_model, args.sampling_timesteps)),
        )
    elif target_model is not None:
        configs.append(
            ("6. + Speculative decode", lambda: run_speculative(model, input_ids, target_model, args.sampling_timesteps)),
        )

    results = {}
    fused_activated = False

    for name, fn in configs:
        # Handle fused block activation marker
        if name == FUSED_MARKER:
            print(f"\n{'='*60}")
            print(f"  Activating fused micro-transformer blocks (Metal/JIT)...")
            print(f"{'='*60}")
            from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks
            swap_to_fused_blocks(model.soft_prompt_generator.transformer)
            swap_to_fused_blocks(model.score_net_head.transformer)
            fused_activated = True
            print("  Done. RMSNorm+FiLM and tiny attention are now fused.")
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Warmup
        print(f"  Warmup ({args.warmup} runs)...")
        for _ in range(args.warmup):
            with torch.no_grad():
                try:
                    fn()
                except Exception as e:
                    print(f"  ERROR during warmup: {e}")
                    break

        # Timed runs
        times = []
        for i in range(args.num_runs):
            with torch.no_grad():
                try:
                    with timed(device) as t:
                        fn()
                    times.append(t['ms'])
                    print(f"  Run {i+1}: {t['ms']:.1f} ms")
                except Exception as e:
                    print(f"  Run {i+1}: ERROR - {e}")
                    import traceback
                    traceback.print_exc()
                    break

        if times:
            import numpy as np
            arr = np.array(times)
            results[name] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'median': float(np.median(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'times': [float(t) for t in times],
            }
            print(f"  => Mean: {results[name]['mean']:.1f} ms  "
                  f"(std={results[name]['std']:.1f}, median={results[name]['median']:.1f})")
        else:
            results[name] = None
            print(f"  => FAILED")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY ({args.num_runs} runs, {args.sampling_timesteps} steps)")
    print(f"{'='*70}")
    print(f"  {'Configuration':<40} {'Mean ms':>10} {'Std':>8} {'Speedup':>8}")
    print(f"  {'-'*66}")

    baseline_mean = None
    for name, stats in results.items():
        if stats is None:
            print(f"  {name:<40} {'FAILED':>10}")
            continue
        if baseline_mean is None:
            baseline_mean = stats['mean']
        speedup = baseline_mean / stats['mean'] if stats['mean'] > 0 else 0
        print(f"  {name:<40} {stats['mean']:>10.1f} {stats['std']:>8.1f} {speedup:>7.2f}x")
    print()

    # Save results to JSON
    import json
    out = {
        'device': str(device),
        'num_runs': args.num_runs,
        'warmup': args.warmup,
        'sampling_timesteps': args.sampling_timesteps,
        'prompt': args.prompt,
        'prompt_tokens': input_ids.shape[1],
        'fused_blocks': fused_activated,
        'results': results,
    }
    out_path = f'results_e2e_{args.sampling_timesteps}steps.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
