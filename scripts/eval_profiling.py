"""
Collect profiling statistics over an evaluation set.

Runs STAR-LDM inference on many prompts and collects per-component timing
distributions (mean, std, median, p95, p99) broken down by prompt length.

Usage:
  # Quick run (20 prompts from C4 validation)
  python -m scripts.eval_profiling --model_path checkpoints/star-ldm --num_prompts 20

  # Full run with varying prompt lengths
  python -m scripts.eval_profiling --model_path checkpoints/star-ldm --num_prompts 100

  # Custom prompts from file (one per line)
  python -m scripts.eval_profiling --model_path checkpoints/star-ldm --prompts_file prompts.txt

  # Save results to JSON
  python -m scripts.eval_profiling --model_path checkpoints/star-ldm --output results.json
"""

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from star_ldm.interface import TransfusionGPTInterface


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class StepProfiler:
    """Collects per-component timing for a single generation."""

    def __init__(self, device):
        self.device = device
        self.records = defaultdict(float)
        self._sync = self._get_sync_fn()

    def _get_sync_fn(self):
        if self.device.type == 'cuda':
            return torch.cuda.synchronize
        elif self.device.type == 'mps':
            return torch.mps.synchronize
        return lambda: None

    @contextmanager
    def track(self, name):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        self.records[name] += (time.perf_counter() - t0) * 1000


def profile_one_generation(model, input_ids, profiler, sampling_timesteps=50, sampler='ddpm'):
    """Profile a single generation, returning per-component timing dict."""
    from star_ldm.models.transfusion import variance_preserving_map, _get_lm_dtype
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    from star_ldm.diffusion.diff_utils import predict_start_from_v, predict_noise_from_v
    from einops import rearrange, repeat

    batch = input_ids.shape[0]
    device = input_ids.device
    var_lambda = 0.2
    sigma2 = 0.05

    sample_noise_schedule = get_scaled_noise_schedule('cosine', scale=3.0)

    times = torch.linspace(1.0, 0., sampling_timesteps + 1, device=device)
    times = repeat(times, 't -> b t', b=batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
    time_pairs = times.unbind(dim=-1)

    z_t = torch.randn((batch, 768), device=device)
    x_start = None

    num_diff = model.num_diffusion_tokens
    diff_mask = torch.zeros((batch, input_ids.shape[1] + num_diff), dtype=torch.bool, device=device)
    diff_mask[:, -num_diff:] = True
    padded_ids = torch.nn.functional.pad(input_ids, (0, num_diff), value=model.tokenizer.pad_token_id)

    for t, t_next in time_pairs:
        with profiler.track("noise_schedule"):
            alpha2 = sample_noise_schedule(t).unsqueeze(-1)
            alpha2_next = sample_noise_schedule(t_next).unsqueeze(-1)

        with profiler.track("soft_prompt_generator"):
            soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)

        with profiler.track("lm_embedding"):
            input_embed = model.lm_embedding(padded_ids).float()
        input_embed[diff_mask] = rearrange(soft_prompt, 'b l d -> (b l) d')

        with profiler.track("gpt2_forward"):
            gpt2_outputs = model.gpt2(
                inputs_embeds=input_embed.to(_get_lm_dtype()),
                labels=None,
                output_hidden_states=True,
            )

        with profiler.track("extract_diffusion_hidden"):
            diffusion_tokens = rearrange(
                gpt2_outputs.hidden_states[-1][diff_mask],
                '(b l) d -> b l d', b=batch, l=num_diff)

        with profiler.track("concat_for_scorenet"):
            diffusion_tokens = torch.cat((soft_prompt, diffusion_tokens), dim=-1)

        with profiler.track("score_net_head"):
            v_pred = model.score_net_head(diffusion_tokens, time_emb)

        with profiler.track("v_to_x0_eps"):
            x_start = predict_start_from_v(z_t, v_pred, alpha2)
            eps = predict_noise_from_v(z_t, v_pred, alpha2)

        if t_next[0] <= 0:
            z_t = x_start
            continue

        with profiler.track("ddpm_step"):
            if sampler == 'ddim':
                z_t = x_start * alpha2_next.sqrt() + eps * (1 - alpha2_next).sqrt()
            else:
                noise = torch.randn_like(z_t)
                alpha2_now = alpha2 / alpha2_next
                min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
                max_var = (1.0 - alpha2_now)
                sigma = torch.exp(var_lambda * torch.log(max_var) + (1 - var_lambda) * torch.log(min_var))
                z_t = 1 / alpha2_now.sqrt() * (z_t - (1 - alpha2_now) / (1 - alpha2).sqrt() * eps) + torch.sqrt(sigma) * noise

    with profiler.track("final_soft_prompt"):
        alpha2_final = torch.full((batch, 1), 1 - sigma2, device=device)
        noised = variance_preserving_map(x_start, alpha2_final)
        soft_prompt, _ = model.soft_prompt_generator(noised, alpha2_final)

    with profiler.track("final_embed_prep"):
        input_embed = model.lm_embedding(input_ids).float()
        input_embed = torch.cat((input_embed, soft_prompt), dim=1)

    with profiler.track("gpt2_generate"):
        gen_ids = model.gpt2.generate(
            inputs_embeds=input_embed.to(_get_lm_dtype()),
            do_sample=True, max_new_tokens=64,
            pad_token_id=model.tokenizer.eos_token_id,
            top_p=0.9, repetition_penalty=1.2,
        )

    generation = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return profiler.records, generation


# ---------------------------------------------------------------------------
# Prompt sources
# ---------------------------------------------------------------------------

def get_c4_prompts(tokenizer, num_prompts=50, max_prefix_len=64, seed=42):
    """Get diverse prompts from C4 validation with varying lengths."""
    from datasets import load_dataset

    ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    prompts = []
    rng = np.random.RandomState(seed)
    for example in ds:
        text = example['text'].strip()
        if len(text) < 50:
            continue
        tokens = tokenizer(text, truncation=True, max_length=128, return_tensors='pt').input_ids[0]
        if len(tokens) < 16:
            continue
        # Random prefix length between 4 and min(max_prefix_len, len-8)
        max_len = min(max_prefix_len, len(tokens) - 8)
        if max_len < 4:
            continue
        prefix_len = rng.randint(4, max_len + 1)
        prefix_ids = tokens[:prefix_len]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        prompts.append({"text": prefix_text, "token_len": prefix_len})
        if len(prompts) >= num_prompts:
            break

    return prompts


def get_fixed_prompts():
    """Hand-crafted prompts of varying lengths for reproducible benchmarking."""
    return [
        {"text": "The", "token_len": 1},
        {"text": "Once upon", "token_len": 2},
        {"text": "The meaning of life is", "token_len": 5},
        {"text": "In the beginning, there was nothing but darkness and silence", "token_len": 10},
        {"text": "The president of the United States announced today that the federal government will be implementing a new policy regarding", "token_len": 20},
        {"text": "Scientists at the research laboratory have been studying the effects of climate change on marine ecosystems for over a decade, and their latest findings suggest that coral reefs in the Pacific Ocean are experiencing unprecedented levels of bleaching due to rising water temperatures and increased ocean acidification", "token_len": 50},
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def print_stats_table(all_results, component_order=None):
    """Print a formatted table of per-component statistics."""
    # Aggregate across all runs
    component_times = defaultdict(list)
    total_times = []

    for run in all_results:
        run_total = sum(run["timings"].values())
        total_times.append(run_total)
        for comp, ms in run["timings"].items():
            component_times[comp].append(ms)

    if component_order is None:
        component_order = sorted(component_times.keys(),
                                 key=lambda c: np.mean(component_times[c]),
                                 reverse=True)

    print(f"\n{'='*100}")
    print(f"{'Component':<30} {'Mean ms':>9} {'Std':>8} {'Median':>9} {'P95':>9} {'P99':>9} {'% Total':>8}")
    print(f"{'='*100}")

    total_mean = np.mean(total_times)
    for comp in component_order:
        times = component_times[comp]
        s = compute_stats(times)
        pct = s['mean'] / total_mean * 100 if total_mean > 0 else 0
        print(f"  {comp:<28} {s['mean']:>9.2f} {s['std']:>8.2f} {s['median']:>9.2f} {s['p95']:>9.2f} {s['p99']:>9.2f} {pct:>7.1f}%")

    total_s = compute_stats(total_times)
    print(f"  {'-'*96}")
    print(f"  {'TOTAL':<28} {total_s['mean']:>9.2f} {total_s['std']:>8.2f} {total_s['median']:>9.2f} {total_s['p95']:>9.2f} {total_s['p99']:>9.2f}")
    print()


def print_by_prompt_length(all_results):
    """Break down timing by prompt token length buckets."""
    buckets = defaultdict(list)
    for run in all_results:
        tlen = run["prompt_token_len"]
        if tlen <= 5:
            bucket = "1-5"
        elif tlen <= 15:
            bucket = "6-15"
        elif tlen <= 30:
            bucket = "16-30"
        else:
            bucket = "31+"
        buckets[bucket].append(sum(run["timings"].values()))

    print(f"\n{'Prompt length':<15} {'N':>5} {'Mean ms':>10} {'Std':>8} {'Median':>10} {'P95':>10}")
    print(f"{'-'*60}")
    for bucket in ["1-5", "6-15", "16-30", "31+"]:
        if bucket in buckets:
            s = compute_stats(buckets[bucket])
            print(f"  {bucket:<13} {s['n']:>5} {s['mean']:>10.1f} {s['std']:>8.1f} {s['median']:>10.1f} {s['p95']:>10.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Eval profiling over many prompts')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num_prompts', type=int, default=50,
                        help='Number of prompts to evaluate')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='File with one prompt per line (overrides C4)')
    parser.add_argument('--eval_set', type=str, default=None,
                        help='Path to eval_prompts.json (e.g. data/eval_prompts.json)')
    parser.add_argument('--use_fixed', action='store_true',
                        help='Use fixed hand-crafted prompts instead of C4')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'])
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup generations to discard')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    interface = TransfusionGPTInterface(model_path=args.model_path, device=str(device))
    model = interface.model
    tokenizer = interface.tokenizer
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get prompts
    if args.eval_set:
        import json as _json
        with open(args.eval_set) as f:
            prompts = _json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.eval_set}")
    elif args.prompts_file:
        with open(args.prompts_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        prompts = []
        for line in lines:
            toks = tokenizer(line, return_tensors='pt').input_ids[0]
            prompts.append({"text": line, "token_len": len(toks)})
    elif args.use_fixed:
        prompts = get_fixed_prompts()
    else:
        print(f"Loading {args.num_prompts} prompts from C4 validation...")
        prompts = get_c4_prompts(tokenizer, num_prompts=args.num_prompts + args.warmup)

    print(f"Collected {len(prompts)} prompts")
    print(f"Token length range: {min(p['token_len'] for p in prompts)}-{max(p['token_len'] for p in prompts)}")

    # Warmup
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} generations)...")
        for i in range(min(args.warmup, len(prompts))):
            input_ids = tokenizer(prompts[i]["text"], return_tensors='pt').input_ids.to(device)
            prof = StepProfiler(device)
            with torch.no_grad():
                profile_one_generation(model, input_ids, prof,
                                       sampling_timesteps=args.sampling_timesteps,
                                       sampler=args.sampler)
        # Remove warmup prompts from eval set
        prompts = prompts[args.warmup:]
        print("Warmup done.\n")

    # Evaluation
    all_results = []
    print(f"Running {len(prompts)} profiled generations ({args.sampling_timesteps} diffusion steps, {args.sampler})...\n")

    for i, prompt_info in enumerate(tqdm(prompts, desc="Evaluating")):
        input_ids = tokenizer(prompt_info["text"], return_tensors='pt').input_ids.to(device)
        prof = StepProfiler(device)

        with torch.no_grad():
            timings, generation = profile_one_generation(
                model, input_ids, prof,
                sampling_timesteps=args.sampling_timesteps,
                sampler=args.sampler)

        result = {
            "prompt": prompt_info["text"],
            "prompt_token_len": prompt_info["token_len"],
            "generation": generation,
            "timings": dict(timings),
            "total_ms": sum(timings.values()),
        }
        all_results.append(result)

    # Print results
    print(f"\n{'='*100}")
    print(f"RESULTS: {len(all_results)} generations, {args.sampling_timesteps} steps, {args.sampler}, device={device}")
    print(f"{'='*100}")

    print_stats_table(all_results)
    print_by_prompt_length(all_results)

    # Per-step averages
    diffusion_components = ['gpt2_forward', 'soft_prompt_generator', 'score_net_head',
                            'ddpm_step', 'noise_schedule', 'v_to_x0_eps',
                            'extract_diffusion_hidden', 'lm_embedding', 'concat_for_scorenet']
    steps = args.sampling_timesteps
    print(f"\nPer diffusion step averages (over {steps} steps × {len(all_results)} runs):")
    for comp in diffusion_components:
        vals = [r["timings"].get(comp, 0) / steps for r in all_results]
        s = compute_stats(vals)
        print(f"  {comp:<30} {s['mean']:>7.3f} ms/step  (std={s['std']:.3f})")

    # Save to JSON
    if args.output:
        output_data = {
            "config": {
                "device": str(device),
                "sampling_timesteps": args.sampling_timesteps,
                "sampler": args.sampler,
                "num_prompts": len(all_results),
                "model_path": args.model_path,
            },
            "aggregate": {},
            "runs": all_results,
        }
        # Aggregate stats
        component_times = defaultdict(list)
        for r in all_results:
            for comp, ms in r["timings"].items():
                component_times[comp].append(ms)
        output_data["aggregate"] = {
            comp: compute_stats(times) for comp, times in component_times.items()
        }
        output_data["aggregate"]["total"] = compute_stats([r["total_ms"] for r in all_results])

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\nDone.")


if __name__ == '__main__':
    main()
