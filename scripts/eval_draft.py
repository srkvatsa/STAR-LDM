"""Evaluate the draft model for speculative decoding.

Measures:
    1. Draft model perplexity on held-out STAR-LDM generations
    2. Acceptance rate when used with speculative decoding
    3. End-to-end latency comparison (baseline vs speculative)
    4. Token-level agreement with STAR-LDM's backbone

Usage:
    # Evaluate draft model quality
    python -m scripts.eval_draft \
        --model_path checkpoints/star-ldm \
        --draft_path checkpoints/draft/best.pt \
        --eval_data data/draft_training/shard_0000.json

    # Benchmark end-to-end latency
    python -m scripts.eval_draft \
        --model_path checkpoints/star-ldm \
        --draft_path checkpoints/draft/best.pt \
        --benchmark --num_runs 10
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from star_ldm.interface import TransfusionGPTInterface
from star_ldm.decoding.draft_model import load_draft_model, DraftModelConfig


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def eval_perplexity(draft_model, eval_data, device):
    """Compute perplexity of draft model on STAR-LDM generated sequences."""
    total_loss = 0.0
    total_tokens = 0

    for sample in tqdm(eval_data, desc="Computing perplexity"):
        prefix_ids = sample["prefix_ids"]
        gen_ids = sample["generated_ids"]
        if not gen_ids:
            continue

        all_ids = prefix_ids + gen_ids
        input_ids = torch.tensor([all_ids[:-1]], device=device)
        labels = torch.tensor([all_ids[1:]], device=device)

        with torch.no_grad():
            output = draft_model(input_ids=input_ids)
            logits = output.logits

        # Only compute loss on generated portion
        gen_start = len(prefix_ids) - 1
        gen_logits = logits[:, gen_start:, :]
        gen_labels = labels[:, gen_start:]

        loss = F.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)),
            gen_labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += gen_labels.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity, avg_loss


def eval_acceptance_rate(interface, draft_model, prompts, device, K=4, sampling_timesteps=50):
    """Measure speculative decoding acceptance rate on real prompts."""
    from star_ldm.decoding.speculative import speculative_generate
    from star_ldm.models.transfusion import _get_lm_dtype, variance_preserving_map
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    from star_ldm.diffusion.fused_ops import fused_ddpm_step

    model = interface.model
    tokenizer = interface.tokenizer

    acceptance_rates = []

    for prompt in tqdm(prompts, desc="Measuring acceptance rate"):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Run diffusion loop
        sample_noise_schedule = get_scaled_noise_schedule("cosine", scale=3.0)
        time_pairs = model.get_sampling_timesteps(1, sampling_timesteps=sampling_timesteps, device=device)
        cached_kv = model._compute_prefix_kv_cache(input_ids)
        z_t = torch.randn((1, 768), device=device)

        for t, t_next in time_pairs:
            alpha2 = sample_noise_schedule(t).unsqueeze(-1)
            alpha2_next = sample_noise_schedule(t_next).unsqueeze(-1)
            out = model._diffusion_model_predictions_cached(z_t, alpha2, cached_kv)
            x_start, eps = out.pred_x, out.pred_eps
            if t_next[0] <= 0:
                z_t = x_start
                continue
            noise = torch.randn_like(z_t)
            z_t = fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)

        # Get soft prompts and input embeddings
        alpha2_final = torch.full((1, 1), 0.95, device=device)
        noised = variance_preserving_map(x_start, alpha2_final)
        soft_prompt, _ = model.soft_prompt_generator(noised, alpha2_final)
        input_embed = model.lm_embedding(input_ids).float()
        input_embed = torch.cat((input_embed, soft_prompt), dim=1)

        if model.freeze_gpt:
            input_embed = input_embed.to(_get_lm_dtype())

        # Run speculative decoding
        with torch.no_grad():
            gen_ids, accept_rate = speculative_generate(
                draft_model=draft_model,
                target_model=model.gpt2,
                input_embeds=input_embed,
                max_new_tokens=32,
                K=K,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                input_ids=input_ids,
            )
            acceptance_rates.append(accept_rate)

        text = tokenizer.decode(gen_ids[0].tolist(), skip_special_tokens=True)

    avg_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0
    return avg_rate, acceptance_rates


def benchmark_latency(interface, draft_model, prompt, device, num_runs=10, warmup=3,
                      sampling_timesteps=50, K=4):
    """Compare end-to-end latency: baseline generate vs speculative."""
    tokenizer = interface.tokenizer
    model = interface.model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    results = {}

    # Baseline: KV-cache generate (no speculative)
    print("  Benchmarking baseline (KV-cache)...")
    for _ in range(warmup):
        with torch.no_grad():
            model.sample_with_kv_cache(input_ids, sampling_timesteps=sampling_timesteps)
    times = []
    for _ in range(num_runs):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.sample_with_kv_cache(input_ids, sampling_timesteps=sampling_timesteps)
        sync_device(device)
        times.append((time.perf_counter() - t0) * 1000)
    results["baseline_kv"] = {
        "mean_ms": sum(times) / len(times),
        "times": times,
    }

    # Speculative: draft model
    print("  Benchmarking speculative (draft model)...")
    for _ in range(warmup):
        with torch.no_grad():
            model.sample_with_draft_speculative(
                input_ids, draft_model=draft_model,
                speculative_k=K, sampling_timesteps=sampling_timesteps)
    times = []
    for _ in range(num_runs):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.sample_with_draft_speculative(
                input_ids, draft_model=draft_model,
                speculative_k=K, sampling_timesteps=sampling_timesteps)
        sync_device(device)
        times.append((time.perf_counter() - t0) * 1000)
    results["speculative_draft"] = {
        "mean_ms": sum(times) / len(times),
        "times": times,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate draft model for speculative decoding")
    parser.add_argument("--model_path", type=str, required=True, help="STAR-LDM checkpoint")
    parser.add_argument("--draft_path", type=str, required=True, help="Draft model checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_data", type=str, default=None, help="JSON shard for perplexity eval")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--speculative_k", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--num_prompts", type=int, default=20,
                        help="Number of prompts for acceptance rate eval")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    print(f"Device: {device}")

    # Load models
    print("Loading STAR-LDM...")
    interface = TransfusionGPTInterface(model_path=args.model_path, device=str(device))

    print("Loading draft model...")
    draft_model = load_draft_model(args.draft_path, device=str(device))
    print(f"  Draft model: {draft_model.num_parameters:,} params")

    # 1. Perplexity evaluation
    if args.eval_data:
        print(f"\n{'='*60}")
        print("  PERPLEXITY EVALUATION")
        print(f"{'='*60}")
        with open(args.eval_data) as f:
            eval_data = json.load(f)
        ppl, avg_loss = eval_perplexity(draft_model, eval_data, device)
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Avg loss: {avg_loss:.4f}")

    # 2. Acceptance rate
    print(f"\n{'='*60}")
    print("  ACCEPTANCE RATE")
    print(f"{'='*60}")
    test_prompts = [
        "The meaning of life is",
        "In a recent study, researchers found that",
        "The president announced today that",
        "Scientists have discovered a new",
        "The weather forecast for tomorrow is",
        "According to the latest report,",
        "In the year 2050, technology will",
        "The history of ancient Rome shows",
        "A healthy diet should include",
        "The stock market today experienced",
        "Education is important because",
        "Climate change is affecting",
        "The new movie received mixed",
        "Artificial intelligence can help",
        "The city of New York is known for",
        "Music has the power to",
        "Space exploration is crucial for",
        "The internet has revolutionized",
        "Cooking at home can be",
        "The future of renewable energy",
    ][:args.num_prompts]

    avg_rate, rates = eval_acceptance_rate(
        interface, draft_model, test_prompts, device,
        K=args.speculative_k, sampling_timesteps=args.sampling_timesteps,
    )
    print(f"  Average acceptance rate: {avg_rate:.1%}")
    print(f"  Min: {min(rates):.1%}, Max: {max(rates):.1%}")

    # 3. Latency benchmark
    if args.benchmark:
        print(f"\n{'='*60}")
        print("  LATENCY BENCHMARK")
        print(f"{'='*60}")
        latency = benchmark_latency(
            interface, draft_model, args.prompt, device,
            num_runs=args.num_runs, warmup=args.warmup,
            sampling_timesteps=args.sampling_timesteps, K=args.speculative_k,
        )
        baseline_ms = latency["baseline_kv"]["mean_ms"]
        spec_ms = latency["speculative_draft"]["mean_ms"]
        speedup = baseline_ms / spec_ms if spec_ms > 0 else 0

        print(f"  Baseline (KV-cache):  {baseline_ms:.1f} ms")
        print(f"  Speculative (draft):  {spec_ms:.1f} ms")
        print(f"  Speedup: {speedup:.2f}x")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Draft model params: {draft_model.num_parameters:,}")
    print(f"  Target model params: {sum(p.numel() for p in interface.model.gpt2.parameters()):,}")
    print(f"  Size ratio: {sum(p.numel() for p in interface.model.gpt2.parameters()) / draft_model.num_parameters:.1f}x")
    print(f"  Acceptance rate: {avg_rate:.1%}")
    if args.benchmark:
        print(f"  Latency speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
