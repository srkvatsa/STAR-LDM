"""Generate training data for the speculative decoding draft model.

Runs full STAR-LDM inference on diverse prefixes and collects:
    - prefix token IDs
    - generated token IDs (from STAR-LDM)
    - (optionally) full logits at each generation step for KL distillation

Data sources:
    - C4 validation set
    - FineWeb sample
    - Custom prompts

Usage:
    # Generate with C4 validation prompts
    python -m scripts.generate_draft_data \
        --model_path checkpoints/star-ldm \
        --output_dir data/draft_training \
        --num_samples 10000 \
        --source c4

    # Generate with custom prefix file (one prompt per line)
    python -m scripts.generate_draft_data \
        --model_path checkpoints/star-ldm \
        --output_dir data/draft_training \
        --prefix_file prompts.txt

    # Save logits for KL distillation (much larger files)
    python -m scripts.generate_draft_data \
        --model_path checkpoints/star-ldm \
        --output_dir data/draft_training \
        --save_logits --num_samples 5000
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from star_ldm.interface import TransfusionGPTInterface


def get_c4_prefixes(num_samples: int, max_prefix_len: int = 64) -> list[str]:
    """Load prefix strings from C4 validation set."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    prefixes = []
    for example in ds:
        text = example["text"].strip()
        # Take first ~max_prefix_len words as prefix
        words = text.split()
        if len(words) < 5:
            continue
        prefix_words = words[:max_prefix_len]
        prefixes.append(" ".join(prefix_words))
        if len(prefixes) >= num_samples:
            break
    return prefixes


def get_fineweb_prefixes(num_samples: int, max_prefix_len: int = 64) -> list[str]:
    """Load prefix strings from FineWeb sample."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    prefixes = []
    for example in ds:
        text = example["text"].strip()
        words = text.split()
        if len(words) < 5:
            continue
        prefix_words = words[:max_prefix_len]
        prefixes.append(" ".join(prefix_words))
        if len(prefixes) >= num_samples:
            break
    return prefixes


def get_file_prefixes(path: str) -> list[str]:
    """Load prefixes from a text file (one per line)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def generate_with_logits(
    interface: TransfusionGPTInterface,
    prompt: str,
    max_new_tokens: int = 32,
    sampling_timesteps: int = 50,
    save_logits: bool = False,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> dict:
    """Run full STAR-LDM and capture generated tokens (and optionally logits).

    Instead of using interface.generate() which only returns text, we need to
    capture the actual token IDs and logits from the GPT-2 generation step.
    """
    model = interface.model
    tokenizer = interface.tokenizer
    device = interface.device

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = input_ids.shape[1]

    # Run diffusion loop with KV-cache
    from star_ldm.models.transfusion import (
        _get_lm_dtype,
        variance_preserving_map,
    )
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    from star_ldm.diffusion.fused_ops import fused_ddpm_step

    sample_noise_schedule = get_scaled_noise_schedule("cosine", scale=3.0)
    batch = 1
    time_pairs = model.get_sampling_timesteps(
        batch, sampling_timesteps=sampling_timesteps, device=device
    )
    cached_kv = model._compute_prefix_kv_cache(input_ids)
    z_t = torch.randn((batch, 768), device=device)

    for time_now, time_next in time_pairs:
        alpha2 = sample_noise_schedule(time_now).unsqueeze(-1)
        alpha2_next = sample_noise_schedule(time_next).unsqueeze(-1)
        model_output = model._diffusion_model_predictions_cached(z_t, alpha2, cached_kv)
        x_start = model_output.pred_x
        eps = model_output.pred_eps
        if time_next[0] <= 0:
            z_t = x_start
            continue
        noise = torch.randn_like(z_t)
        z_t = fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)

    # Compute final soft prompts
    sigma2 = 0.05
    alpha2_final = torch.full((batch, 1), 1 - sigma2, device=device)
    noised = variance_preserving_map(x_start, alpha2_final)
    soft_prompt, _ = model.soft_prompt_generator(noised, alpha2_final)

    # Build full input embeddings
    input_embed = model.lm_embedding(input_ids).float()
    input_embed = torch.cat((input_embed, soft_prompt), dim=1)
    if model.freeze_gpt:
        input_embed = input_embed.to(_get_lm_dtype())

    # Manual autoregressive generation to capture logits
    generated_ids = []
    all_logits = [] if save_logits else None

    # Initialize KV-cache from prefix + soft prompt
    out = model.gpt2(inputs_embeds=input_embed, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]  # (1, V)

    for step in range(max_new_tokens):
        logits = next_logits.float()

        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for prev_id in set(generated_ids):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= repetition_penalty
                else:
                    logits[0, prev_id] *= repetition_penalty

        # Top-p filtering
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            idx_to_remove = remove_mask.scatter(-1, sorted_idx, remove_mask)
            logits = logits.masked_fill(idx_to_remove, float("-inf"))

        if save_logits:
            all_logits.append(logits.squeeze(0).cpu().half())

        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_id = token.squeeze().item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        # Feed token back
        token_embed = model.lm_embedding(token)
        if model.freeze_gpt:
            token_embed = token_embed.to(_get_lm_dtype())
        out = model.gpt2(inputs_embeds=token_embed, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    result = {
        "prefix": prompt,
        "prefix_ids": input_ids.squeeze(0).cpu().tolist(),
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }
    if save_logits and all_logits:
        result["logits"] = torch.stack(all_logits)  # (num_generated, V), fp16

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate draft model training data")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--source", type=str, default="fineweb", choices=["c4", "fineweb"],
                        help="Prefix source. Use 'fineweb' (default) to match STAR-LDM's "
                             "training distribution. Use 'c4' only for held-out evaluation.")
    parser.add_argument("--prefix_file", type=str, default=None,
                        help="Text file with one prompt per line, or .json eval_prompts file")
    parser.add_argument("--eval_prompts", type=str, default=None,
                        help="Path to eval_prompts.json (uses 'text' field as prefixes)")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--save_logits", action="store_true",
                        help="Save full logits for KL distillation (large files)")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of samples per shard file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prefixes
    print("Loading prefixes...")
    if args.eval_prompts:
        with open(args.eval_prompts) as f:
            prompts_data = json.load(f)
        prefixes = [p["text"] for p in prompts_data]
        print(f"  (from eval_prompts.json, {len(prefixes)} prompts)")
    elif args.prefix_file:
        if args.prefix_file.endswith(".json"):
            with open(args.prefix_file) as f:
                data = json.load(f)
            if isinstance(data, list) and isinstance(data[0], dict) and "text" in data[0]:
                prefixes = [p["text"] for p in data]
            else:
                prefixes = data
        else:
            prefixes = get_file_prefixes(args.prefix_file)
    elif args.source == "c4":
        prefixes = get_c4_prefixes(args.num_samples)
    else:
        prefixes = get_fineweb_prefixes(args.num_samples)

    print(f"Loaded {len(prefixes)} prefixes")

    # Load STAR-LDM
    print("Loading STAR-LDM...")
    interface = TransfusionGPTInterface(model_path=args.model_path, device=args.device)

    # Generate
    shard_idx = 0
    shard_data = []
    shard_logits = []
    stats = {"total": 0, "avg_gen_len": 0.0, "total_time": 0.0}

    for i, prefix in enumerate(tqdm(prefixes, desc="Generating")):
        t0 = time.perf_counter()
        try:
            result = generate_with_logits(
                interface,
                prefix,
                max_new_tokens=args.max_new_tokens,
                sampling_timesteps=args.sampling_timesteps,
                save_logits=args.save_logits,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

        elapsed = time.perf_counter() - t0
        stats["total"] += 1
        stats["avg_gen_len"] += len(result["generated_ids"])
        stats["total_time"] += elapsed

        # Separate logits from JSON-serializable data
        logits_tensor = result.pop("logits", None)
        shard_data.append(result)
        if logits_tensor is not None:
            shard_logits.append(logits_tensor)

        # Save shard
        if len(shard_data) >= args.shard_size:
            _save_shard(args.output_dir, shard_idx, shard_data, shard_logits, args.save_logits)
            shard_idx += 1
            shard_data = []
            shard_logits = []

    # Save final partial shard
    if shard_data:
        _save_shard(args.output_dir, shard_idx, shard_data, shard_logits, args.save_logits)

    # Save stats
    if stats["total"] > 0:
        stats["avg_gen_len"] /= stats["total"]
        stats["avg_time_per_sample"] = stats["total_time"] / stats["total"]

    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone! Generated {stats['total']} samples")
    print(f"  Avg generation length: {stats['avg_gen_len']:.1f} tokens")
    print(f"  Avg time per sample: {stats.get('avg_time_per_sample', 0):.2f}s")
    print(f"  Saved to: {args.output_dir}")


def _save_shard(output_dir, shard_idx, data, logits, save_logits):
    """Save a shard of data to disk."""
    json_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.json")
    with open(json_path, "w") as f:
        json.dump(data, f)

    if save_logits and logits:
        logits_path = os.path.join(output_dir, f"shard_{shard_idx:04d}_logits.pt")
        torch.save(logits, logits_path)

    print(f"  Saved shard {shard_idx} ({len(data)} samples)")


if __name__ == "__main__":
    main()
