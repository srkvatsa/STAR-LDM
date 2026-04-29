"""Profile GPT-2 Large decode-8 attention pattern for kernel optimization.

STAR-LDM runs GPT-2 Large on 8 soft-prompt tokens with a KV-cached prefix
at every diffusion step. This is the dominant cost (~68% of each step).

This script profiles:
  1. Layer scaling: how time grows with number of transformer layers
  2. Op-level breakdown: c_attn, SDPA, c_proj, MLP, LayerNorm per layer
  3. Full diffusion step breakdown: SoftPromptGen vs GPT-2 vs ScoreNetHead
  4. Impact analysis: what speedup on GPT-2 means for end-to-end

The decode-8 pattern is unusual: Q(1,20,8,64), K/V(1,20,prefix+8,64).
Standard SDPA/FlashAttention is overbuilt for this tiny query count.
A fused kernel could combine c_attn + SDPA + c_proj per layer, or even
fuse across multiple layers.

Usage:
  python -m scripts.profile_decode8 --model_path checkpoints/star-ldm
  python -m scripts.profile_decode8 --dummy
"""

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timed(device):
    sync(device)
    t0 = time.perf_counter()
    result = {}
    yield result
    sync(device)
    result["ms"] = (time.perf_counter() - t0) * 1000


def create_model(device, model_path=None):
    if model_path:
        from star_ldm.interface import TransfusionGPTInterface
        interface = TransfusionGPTInterface(model_path=model_path, device=str(device))
        return interface.model
    cfg = OmegaConf.create({
        "dataset_name": "fineweb_100b",
        "train": {"freeze_gpt": True, "lm_name": "gpt2-large", "global_norm": True},
        "sampling": {"noise_schedule_name": "cosine", "noise_schedule_scale": 1.0},
        "diffusion_loss": {
            "weighting_name": "sigmoid",
            "weighting_kwargs": {"gamma_shift": 0.0},
            "train_schedule": "cosine",
            "cosine_shift": 0.0,
        },
        "prompt_generator": {"dim": 1024, "dim_head": 64, "depth": 6, "prompt_length": 8, "dropout": 0.0},
        "scorenet_head": {"dim": 1024, "depth": 6, "dropout": 0.0, "output_dim_mult": 4},
    })
    from star_ldm.models.transfusion import TransfusionGPT
    model = TransfusionGPT(
        dataset_name="fineweb_100b", transfusion_cfg=cfg,
        gpt2_model_name="gpt2-large",
        gamma_min=-15, gamma_max=15,
        clf_guidance_dropout=0.1, scale_by_std=True, global_norm=True,
    )
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# 1. Layer scaling benchmark
# ---------------------------------------------------------------------------

def benchmark_layer_scaling(model, device, prefix_len=5, num_query_tokens=8, warmup=5, runs=20):
    """Measure GPT-2 forward time as a function of number of layers.

    We manually run hidden states through individual transformer blocks,
    passing layer-specific past_key_values as a tuple (key, value).
    """
    from star_ldm.models.transfusion import _get_lm_dtype

    gpt2 = model.gpt2
    tokenizer = model.tokenizer
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Build full KV cache from prefix
    with torch.no_grad():
        prefix_out = gpt2(input_ids, use_cache=True)
    full_kv = prefix_out.past_key_values

    n_layers = gpt2.config.n_layer  # 36 for gpt2-large

    # Query tokens (8 random embeddings, like soft prompts) — must be float32 for GPT-2
    query_embed = torch.randn(1, num_query_tokens, gpt2.config.n_embd, device=device, dtype=torch.float32)

    layer_counts = [1, 2, 4, 6, 8, 12, 18, 24, 36]
    layer_counts = [l for l in layer_counts if l <= n_layers]

    print(f"\n{'='*70}")
    print(f"  LAYER SCALING: {num_query_tokens} query tokens, prefix_len={prefix_len}")
    print(f"  GPT-2 Large: {n_layers} layers, {gpt2.config.n_embd}d, {gpt2.config.n_head} heads")
    print(f"{'='*70}")
    print(f"  {'Layers':>8} {'Mean ms':>10} {'Std ms':>10} {'ms/layer':>10}")
    print(f"  {'-'*40}")

    import numpy as np
    results = {}
    for n_lay in layer_counts:
        # Each block indexes into full_kv by its own layer_idx
        def run_layers(n=n_lay):
            h = query_embed.clone()
            for i in range(n):
                block = gpt2.transformer.h[i]
                out = block(h, past_key_values=full_kv, use_cache=False)
                h = out[0]
                # Block may squeeze batch dim; restore if needed
                if h.dim() == 2:
                    h = h.unsqueeze(0)
            return h

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                run_layers()

        # Timed runs
        times = []
        for _ in range(runs):
            with timed(device) as t:
                with torch.no_grad():
                    run_layers()
            times.append(t["ms"])

        arr = np.array(times)
        results[n_lay] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"  {n_lay:>8} {arr.mean():>10.2f} {arr.std():>10.3f} {arr.mean()/n_lay:>10.3f}")

    return results


# ---------------------------------------------------------------------------
# 2. Op-level breakdown within a single layer
# ---------------------------------------------------------------------------

def benchmark_op_breakdown(model, device, prefix_len=5, num_query_tokens=8, warmup=10, runs=30):
    """Break down time within a single GPT-2 layer for 8-token decode."""
    from star_ldm.models.transfusion import _get_lm_dtype

    gpt2 = model.gpt2
    tokenizer = model.tokenizer
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        prefix_out = gpt2(input_ids, use_cache=True)
    full_kv = prefix_out.past_key_values

    # Examine layer 0 as representative
    block = gpt2.transformer.h[0]
    layer_kv = full_kv.layers[0]
    past_key = layer_kv.keys    # (1, n_head, prefix_len, head_dim)
    past_value = layer_kv.values

    hidden = torch.randn(1, num_query_tokens, gpt2.config.n_embd, device=device, dtype=torch.float32)
    n_head = gpt2.config.n_head
    head_dim = gpt2.config.n_embd // n_head

    print(f"\n{'='*70}")
    print(f"  OP-LEVEL BREAKDOWN: Single GPT-2 layer, {num_query_tokens} query tokens")
    print(f"  KV shape: K={past_key.shape}, V={past_value.shape}")
    print(f"{'='*70}")

    ops = {}

    # --- LayerNorm (attn) ---
    ln_1 = block.ln_1
    for _ in range(warmup):
        with torch.no_grad():
            ln_1(hidden)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                normed = ln_1(hidden)
        times.append(t["ms"])
    ops["ln_1 (LayerNorm)"] = times

    # --- c_attn projection (QKV) ---
    c_attn = block.attn.c_attn
    for _ in range(warmup):
        with torch.no_grad():
            c_attn(normed)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                qkv = c_attn(normed)
        times.append(t["ms"])
    ops["c_attn (QKV proj)"] = times

    # --- Split Q, K, V and reshape ---
    q, k_new, v_new = qkv.split(gpt2.config.n_embd, dim=-1)
    q = q.view(1, num_query_tokens, n_head, head_dim).transpose(1, 2)
    k_new = k_new.view(1, num_query_tokens, n_head, head_dim).transpose(1, 2)
    v_new = v_new.view(1, num_query_tokens, n_head, head_dim).transpose(1, 2)

    # Concat with past KV
    k_full = torch.cat([past_key, k_new], dim=2)
    v_full = torch.cat([past_value, v_new], dim=2)

    # --- SDPA ---
    for _ in range(warmup):
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k_full, v_full, is_causal=False
                )
        times.append(t["ms"])
    ops["SDPA (attention)"] = times

    # --- c_proj ---
    attn_flat = attn_out.transpose(1, 2).reshape(1, num_query_tokens, gpt2.config.n_embd)
    c_proj = block.attn.c_proj
    for _ in range(warmup):
        with torch.no_grad():
            c_proj(attn_flat)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                proj_out = c_proj(attn_flat)
        times.append(t["ms"])
    ops["c_proj (out proj)"] = times

    # --- LayerNorm (MLP) ---
    ln_2 = block.ln_2
    residual = hidden + proj_out
    for _ in range(warmup):
        with torch.no_grad():
            ln_2(residual)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                mlp_normed = ln_2(residual)
        times.append(t["ms"])
    ops["ln_2 (LayerNorm)"] = times

    # --- MLP (fc + act + proj) ---
    mlp = block.mlp
    for _ in range(warmup):
        with torch.no_grad():
            mlp(mlp_normed)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                mlp_out = mlp(mlp_normed)
        times.append(t["ms"])
    ops["MLP (fc+gelu+proj)"] = times

    # Print results
    import numpy as np
    print(f"  {'Operation':<25} {'Mean ms':>10} {'Std ms':>10} {'% Layer':>8}")
    print(f"  {'-'*55}")
    total = 0
    op_means = {}
    for name, ts in ops.items():
        arr = np.array(ts)
        op_means[name] = arr.mean()
        total += arr.mean()

    for name, ts in ops.items():
        arr = np.array(ts)
        pct = arr.mean() / total * 100 if total > 0 else 0
        print(f"  {name:<25} {arr.mean():>10.4f} {arr.std():>10.4f} {pct:>7.1f}%")
    print(f"  {'-'*55}")
    print(f"  {'TOTAL':<25} {total:>10.4f}")

    # Compute theoretical single-layer time vs actual
    print(f"\n  Sum of individual ops:   {total:.4f} ms")

    # Also time the full block as one unit
    for _ in range(warmup):
        with torch.no_grad():
            block(hidden, past_key_values=full_kv, use_cache=False)
    times = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                block(hidden, past_key_values=full_kv, use_cache=False)
        times.append(t["ms"])
    actual = np.mean(times)
    print(f"  Actual block forward:   {actual:.4f} ms")
    print(f"  Overhead (dispatch etc): {total - actual:.4f} ms ({(total-actual)/actual*100:.1f}%)")
    print(f"    (Negative = MPS pipelining hides latency)")

    return ops, op_means


# ---------------------------------------------------------------------------
# 3. Full diffusion step breakdown
# ---------------------------------------------------------------------------

def benchmark_diffusion_step(model, device, warmup=3, runs=10, sampling_timesteps=50):
    """Break down a full STAR-LDM sampling run into components."""
    from star_ldm.models.transfusion import _get_lm_dtype, variance_preserving_map
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    from star_ldm.diffusion.fused_ops import fused_ddpm_step
    from einops import rearrange, repeat

    tokenizer = model.tokenizer
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    batch = 1

    sample_noise_schedule = get_scaled_noise_schedule("cosine", scale=3.0)
    times_sched = torch.linspace(1.0, 0.0, sampling_timesteps + 1, device=device)
    times_sched = repeat(times_sched, "t -> b t", b=batch)
    times_sched = torch.stack((times_sched[:, :-1], times_sched[:, 1:]), dim=0)
    time_pairs = times_sched.unbind(dim=-1)

    num_diff = model.num_diffusion_tokens
    diff_mask = torch.zeros((batch, input_ids.shape[1] + num_diff), dtype=torch.bool, device=device)
    diff_mask[:, -num_diff:] = True
    padded_ids = torch.nn.functional.pad(input_ids, (0, num_diff), value=tokenizer.pad_token_id)

    print(f"\n{'='*70}")
    print(f"  FULL DIFFUSION STEP BREAKDOWN ({sampling_timesteps} steps)")
    print(f"{'='*70}")

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            z_t = torch.randn((batch, 768), device=device)
            for t, t_next in time_pairs:
                alpha2 = sample_noise_schedule(t).unsqueeze(-1)
                alpha2_next = sample_noise_schedule(t_next).unsqueeze(-1)
                soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)
                input_embed = model.lm_embedding(padded_ids).float()
                input_embed[diff_mask] = rearrange(soft_prompt, "b l d -> (b l) d")
                gpt2_out = model.gpt2(inputs_embeds=input_embed.to(_get_lm_dtype()),
                                       output_hidden_states=True)
                diff_hidden = rearrange(
                    gpt2_out.hidden_states[-1][diff_mask],
                    "(b l) d -> b l d", b=batch, l=num_diff
                )
                diff_hidden = torch.cat((soft_prompt, diff_hidden), dim=-1)
                v_pred = model.score_net_head(diff_hidden, time_emb)
                from star_ldm.diffusion.diff_utils import predict_start_from_v, predict_noise_from_v
                x_start = predict_start_from_v(z_t, v_pred, alpha2)
                eps = predict_noise_from_v(z_t, v_pred, alpha2)
                if t_next[0] <= 0:
                    z_t = x_start
                    continue
                noise = torch.randn_like(z_t)
                z_t = fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)

    # Timed run: accumulate per-component times across all steps
    component_times = defaultdict(list)

    for run_idx in range(runs):
        z_t = torch.randn((batch, 768), device=device)
        x_start = None

        with timed(device) as total_t:
            with torch.no_grad():
                for step_idx, (t, t_next) in enumerate(time_pairs):
                    alpha2 = sample_noise_schedule(t).unsqueeze(-1)
                    alpha2_next = sample_noise_schedule(t_next).unsqueeze(-1)

                    with timed(device) as spg_t:
                        soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)
                    component_times["soft_prompt_gen"].append(spg_t["ms"])

                    with timed(device) as emb_t:
                        input_embed = model.lm_embedding(padded_ids).float()
                        input_embed[diff_mask] = rearrange(soft_prompt, "b l d -> (b l) d")
                    component_times["embed_prep"].append(emb_t["ms"])

                    with timed(device) as gpt_t:
                        gpt2_out = model.gpt2(
                            inputs_embeds=input_embed.to(_get_lm_dtype()),
                            output_hidden_states=True,
                        )
                    component_times["gpt2_forward"].append(gpt_t["ms"])

                    with timed(device) as score_t:
                        diff_hidden = rearrange(
                            gpt2_out.hidden_states[-1][diff_mask],
                            "(b l) d -> b l d", b=batch, l=num_diff
                        )
                        diff_hidden = torch.cat((soft_prompt, diff_hidden), dim=-1)
                        v_pred = model.score_net_head(diff_hidden, time_emb)
                    component_times["score_net_head"].append(score_t["ms"])

                    with timed(device) as diff_t:
                        from star_ldm.diffusion.diff_utils import predict_start_from_v, predict_noise_from_v
                        x_start = predict_start_from_v(z_t, v_pred, alpha2)
                        eps = predict_noise_from_v(z_t, v_pred, alpha2)
                        if t_next[0] <= 0:
                            z_t = x_start
                        else:
                            noise = torch.randn_like(z_t)
                            z_t = fused_ddpm_step(z_t, eps, noise, alpha2, alpha2_next, 0.2)
                    component_times["ddpm_step"].append(diff_t["ms"])

        component_times["total_loop"].append(total_t["ms"])

        # Final generation
        with timed(device) as gen_t:
            with torch.no_grad():
                alpha2_final = torch.full((batch, 1), 0.95, device=device)
                noised = variance_preserving_map(x_start, alpha2_final)
                soft_prompt, _ = model.soft_prompt_generator(noised, alpha2_final)
                input_embed = model.lm_embedding(input_ids).float()
                input_embed = torch.cat((input_embed, soft_prompt), dim=1)
                gen_ids = model.gpt2.generate(
                    inputs_embeds=input_embed.to(_get_lm_dtype()),
                    do_sample=True, max_new_tokens=32,
                    pad_token_id=tokenizer.eos_token_id,
                    top_p=0.9, repetition_penalty=1.2,
                )
        component_times["ar_generation"].append(gen_t["ms"])

    # Print results
    import numpy as np
    print(f"\n  {'Component':<25} {'Per-step ms':>12} {'Total ms':>12} {'% E2E':>8}")
    print(f"  {'-'*60}")

    total_e2e = np.mean(component_times["total_loop"]) + np.mean(component_times["ar_generation"])

    for name, label, is_per_step in [
        ("soft_prompt_gen", "SoftPromptGen", True),
        ("embed_prep", "Embed prep", True),
        ("gpt2_forward", "GPT-2 forward", True),
        ("score_net_head", "ScoreNetHead", True),
        ("ddpm_step", "DDPM step", True),
        ("ar_generation", "AR generation", False),
    ]:
        arr = np.array(component_times[name])
        if is_per_step:
            per_step = arr.mean()
            total = per_step * sampling_timesteps
        else:
            per_step = arr.mean()
            total = per_step
        pct = total / total_e2e * 100
        if is_per_step:
            print(f"  {label:<25} {per_step:>12.3f} {total:>12.1f} {pct:>7.1f}%")
        else:
            print(f"  {label:<25} {'—':>12} {total:>12.1f} {pct:>7.1f}%")

    print(f"  {'-'*60}")
    print(f"  {'End-to-end':<25} {'':>12} {total_e2e:>12.1f}")

    # Impact analysis
    gpt2_per_step = np.mean(component_times["gpt2_forward"])
    gpt2_total = gpt2_per_step * sampling_timesteps
    gpt2_pct = gpt2_total / total_e2e * 100

    print(f"\n  GPT-2 forward: {gpt2_per_step:.2f} ms/step x {sampling_timesteps} steps = {gpt2_total:.0f} ms ({gpt2_pct:.0f}% of E2E)")
    print(f"\n  Impact of fused GPT-2 decode-8 kernel:")
    for speedup in [1.25, 1.5, 2.0, 3.0, 5.0]:
        new_gpt2 = gpt2_total / speedup
        new_total = total_e2e - gpt2_total + new_gpt2
        e2e_speedup = total_e2e / new_total
        saved_ms = total_e2e - new_total
        print(f"    {speedup:.1f}x GPT-2 speedup → {e2e_speedup:.2f}x E2E speedup ({saved_ms:.0f} ms saved)")

    return component_times


# ---------------------------------------------------------------------------
# 4. KV-cache vs no-cache comparison for 8 tokens
# ---------------------------------------------------------------------------

def benchmark_kv_vs_nocache(model, device, warmup=5, runs=20):
    """Compare GPT-2 forward with and without KV cache for 8 tokens."""
    from star_ldm.models.transfusion import _get_lm_dtype

    gpt2 = model.gpt2
    tokenizer = model.tokenizer
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = input_ids.shape[1]

    print(f"\n{'='*70}")
    print(f"  KV-CACHE vs NO-CACHE for decode-8")
    print(f"  Prefix: {prefix_len} tokens, Query: 8 soft-prompt tokens")
    print(f"{'='*70}")

    # No cache: full forward on prefix + 8 tokens
    soft_tokens = torch.randn(1, 8, gpt2.config.n_embd, device=device, dtype=_get_lm_dtype())
    prefix_embed = gpt2.transformer.wte(input_ids).to(_get_lm_dtype())
    full_embed = torch.cat([prefix_embed, soft_tokens], dim=1)

    for _ in range(warmup):
        with torch.no_grad():
            gpt2(inputs_embeds=full_embed, output_hidden_states=True)
    times_nocache = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                gpt2(inputs_embeds=full_embed, output_hidden_states=True)
        times_nocache.append(t["ms"])

    # With cache: prefix cached, forward only 8 tokens
    with torch.no_grad():
        prefix_out = gpt2(inputs_embeds=prefix_embed, use_cache=True)
    kv_cache = prefix_out.past_key_values

    for _ in range(warmup):
        with torch.no_grad():
            gpt2(inputs_embeds=soft_tokens, past_key_values=kv_cache, use_cache=False,
                  output_hidden_states=True)
    times_cached = []
    for _ in range(runs):
        with timed(device) as t:
            with torch.no_grad():
                gpt2(inputs_embeds=soft_tokens, past_key_values=kv_cache, use_cache=False,
                      output_hidden_states=True)
        times_cached.append(t["ms"])

    import numpy as np
    nc = np.array(times_nocache)
    c = np.array(times_cached)
    print(f"  No cache (prefix+8):  {nc.mean():.2f} ms (std={nc.std():.2f})")
    print(f"  KV-cached (8 only):   {c.mean():.2f} ms (std={c.std():.2f})")
    print(f"  Speedup from cache:   {nc.mean()/c.mean():.2f}x")
    print(f"  Saved:                {nc.mean()-c.mean():.2f} ms/step = {(nc.mean()-c.mean())*50:.0f} ms total")


# ---------------------------------------------------------------------------
# 5. Attention pattern analysis
# ---------------------------------------------------------------------------

def analyze_attention_shapes(model, device):
    """Print the exact tensor shapes flowing through GPT-2 decode-8."""
    from star_ldm.models.transfusion import _get_lm_dtype

    gpt2 = model.gpt2
    tokenizer = model.tokenizer
    prompt = "The meaning of life is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = input_ids.shape[1]

    with torch.no_grad():
        prefix_out = gpt2(input_ids, use_cache=True)
    kv = prefix_out.past_key_values

    n_head = gpt2.config.n_head
    head_dim = gpt2.config.n_embd // n_head
    n_layers = gpt2.config.n_layer
    kv_len = kv.layers[0].keys.shape[2]

    print(f"\n{'='*70}")
    print(f"  ATTENTION SHAPE ANALYSIS: GPT-2 Large decode-8")
    print(f"{'='*70}")
    print(f"  Model:     GPT-2 Large ({n_layers} layers, {gpt2.config.n_embd}d)")
    print(f"  Heads:     {n_head} x {head_dim}d")
    print(f"  Prefix:    {prefix_len} tokens → KV cache len = {kv_len}")
    print(f"  Query:     8 soft-prompt tokens")
    print(f"")
    print(f"  Per-layer attention shapes:")
    print(f"    Q:  (1, {n_head}, 8, {head_dim})")
    print(f"    K:  (1, {n_head}, {kv_len}+8, {head_dim})  = (1, {n_head}, {kv_len+8}, {head_dim})")
    print(f"    V:  (1, {n_head}, {kv_len}+8, {head_dim})  = (1, {n_head}, {kv_len+8}, {head_dim})")
    print(f"    Attn weights: (1, {n_head}, 8, {kv_len+8})")
    print(f"    Output: (1, {n_head}, 8, {head_dim})")
    print(f"")
    print(f"  FLOPs per layer (attention only):")
    seq_kv = kv_len + 8
    qk_flops = 2 * n_head * 8 * seq_kv * head_dim
    av_flops = 2 * n_head * 8 * head_dim * seq_kv
    total_attn_flops = qk_flops + av_flops
    print(f"    QK^T:     2 * {n_head} * 8 * {seq_kv} * {head_dim} = {qk_flops:,}")
    print(f"    Attn@V:   2 * {n_head} * 8 * {head_dim} * {seq_kv} = {av_flops:,}")
    print(f"    Total:    {total_attn_flops:,} ({total_attn_flops/1e6:.2f} MFLOPs)")
    print(f"    All {n_layers} layers: {total_attn_flops*n_layers:,} ({total_attn_flops*n_layers/1e6:.1f} MFLOPs)")
    print(f"")
    print(f"  FLOPs per layer (projections):")
    c_attn_flops = 2 * 8 * gpt2.config.n_embd * 3 * gpt2.config.n_embd  # QKV projection
    c_proj_flops = 2 * 8 * gpt2.config.n_embd * gpt2.config.n_embd
    mlp_flops = 2 * 8 * gpt2.config.n_embd * 4 * gpt2.config.n_embd * 2  # fc1 + fc2
    total_layer_flops = total_attn_flops + c_attn_flops + c_proj_flops + mlp_flops
    print(f"    c_attn (QKV):  {c_attn_flops:,} ({c_attn_flops/1e6:.1f} MFLOPs)")
    print(f"    c_proj (out):  {c_proj_flops:,} ({c_proj_flops/1e6:.1f} MFLOPs)")
    print(f"    MLP (fc1+fc2): {mlp_flops:,} ({mlp_flops/1e6:.1f} MFLOPs)")
    print(f"    Layer total:   {total_layer_flops:,} ({total_layer_flops/1e6:.1f} MFLOPs)")
    print(f"    All {n_layers} layers:  {total_layer_flops*n_layers/1e9:.2f} GFLOPs")
    print(f"")

    # Compute arithmetic intensity
    # Memory: read Q(8*d) + K(seq*d) + V(seq*d) per head, write O(8*d)
    bytes_per_elem = 2  # fp16
    mem_per_head = (8 * head_dim + 2 * seq_kv * head_dim + 8 * head_dim) * bytes_per_elem
    mem_all_heads = mem_per_head * n_head
    total_mem = mem_all_heads + c_attn_flops // 2 * bytes_per_elem  # rough
    ai = total_attn_flops / mem_all_heads
    print(f"  Arithmetic intensity (attn only): {ai:.2f} FLOPs/byte")
    print(f"    → Memory-bound! (Apple M4 Max roofline: ~200 FLOPs/byte for compute-bound)")
    print(f"")
    print(f"  Kernel fusion opportunities:")
    print(f"    1. Fuse c_attn + SDPA + c_proj per layer (eliminate intermediate writes)")
    print(f"    2. Fuse LayerNorm into attention kernel (read hidden once)")
    print(f"    3. Multi-layer fusion: process 2-4 layers in one kernel launch")
    print(f"    4. Custom tiny-SDPA: 8-query specialized (no block tiling needed)")
    print(f"    5. Persistent kernel: keep KV-cache in registers/threadgroup memory")


def main():
    parser = argparse.ArgumentParser(description="Profile GPT-2 decode-8 attention")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--skip_diffusion", action="store_true",
                        help="Skip the full diffusion step breakdown (slow)")
    args = parser.parse_args()

    if not args.dummy and not args.model_path:
        parser.error("Provide --model_path or --dummy")

    device = get_device()
    print(f"Device: {device}")

    print("Loading model...")
    model = create_model(device, args.model_path if not args.dummy else None)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        # Shape analysis (no timing needed)
        analyze_attention_shapes(model, device)

        # KV-cache vs no-cache
        benchmark_kv_vs_nocache(model, device, warmup=args.warmup, runs=args.runs)

        # Op-level breakdown
        benchmark_op_breakdown(model, device, warmup=args.warmup, runs=args.runs)

        # Layer scaling
        benchmark_layer_scaling(model, device, warmup=args.warmup, runs=args.runs)

        # Full diffusion step breakdown (slowest, optional)
        if not args.skip_diffusion:
            benchmark_diffusion_step(model, device, warmup=2, runs=5,
                                     sampling_timesteps=args.sampling_timesteps)


if __name__ == "__main__":
    main()
