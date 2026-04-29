"""
Profile STAR-LDM inference to identify bottlenecks.

Produces:
  1. Per-component timing table (stdout)
  2. Chrome trace file (open in chrome://tracing)
  3. Summary of kernel launch counts and time per diffusion step

Usage:
  python -m scripts.profile_inference --model_path checkpoints/star-ldm

On Apple Silicon (no checkpoint needed — uses random weights for profiling):
  python -m scripts.profile_inference --dummy
"""

import argparse
import time
import json
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Manual timing instrumentation (works on all devices including MPS)
# ---------------------------------------------------------------------------

class ManualProfiler:
    """Simple timer that wraps model components with hooks.

    We can't rely on torch.profiler for MPS — it doesn't support
    MPS activity tracing. So we use manual synchronization + time.perf_counter.
    """

    def __init__(self, device):
        self.device = device
        self.records = defaultdict(list)
        self._sync = self._get_sync_fn()

    def _get_sync_fn(self):
        if self.device.type == 'cuda':
            return torch.cuda.synchronize
        elif self.device.type == 'mps':
            return torch.mps.synchronize
        else:
            return lambda: None

    @contextmanager
    def track(self, name):
        self._sync()
        t0 = time.perf_counter()
        yield
        self._sync()
        t1 = time.perf_counter()
        self.records[name].append((t1 - t0) * 1000)  # ms

    def summary(self):
        print("\n" + "=" * 80)
        print(f"{'Component':<45} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10} {'% Total':>8}")
        print("=" * 80)
        total = sum(sum(v) for v in self.records.values())
        # Sort by total time descending
        sorted_items = sorted(self.records.items(), key=lambda x: sum(x[1]), reverse=True)
        for name, times in sorted_items:
            total_ms = sum(times)
            avg_ms = total_ms / len(times)
            pct = (total_ms / total * 100) if total > 0 else 0
            print(f"  {name:<43} {len(times):>6} {total_ms:>10.2f} {avg_ms:>10.3f} {pct:>7.1f}%")
        print("-" * 80)
        print(f"  {'TOTAL':<43} {'':>6} {total:>10.2f}")
        print()
        return {name: {"calls": len(times), "total_ms": sum(times), "avg_ms": sum(times)/len(times)}
                for name, times in sorted_items}


# ---------------------------------------------------------------------------
# Instrumented sampling — hooks into the real model methods
# ---------------------------------------------------------------------------

def profile_sampling(model, input_ids, profiler, sampling_timesteps=50,
                     sampler='ddpm', cls_free_guidance=1.0):
    """Run the sampling loop with manual instrumentation around each component."""
    from star_ldm.models.transfusion import variance_preserving_map
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    from einops import rearrange, repeat
    from tqdm import tqdm

    batch = input_ids.shape[0]
    device = input_ids.device
    var_lambda = 0.2
    sigma2 = 0.05
    cosine_scale = 3.0

    sample_noise_schedule = get_scaled_noise_schedule('cosine', scale=cosine_scale)

    # Build time pairs
    times = torch.linspace(1.0, 0., sampling_timesteps + 1, device=device)
    times = repeat(times, 't -> b t', b=batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
    time_pairs = times.unbind(dim=-1)

    # Start from noise
    z_t = torch.randn((batch, 768), device=device)
    x_start = None

    # Pad input_ids for diffusion tokens
    num_diff = model.num_diffusion_tokens
    diff_mask = torch.zeros((batch, input_ids.shape[1] + num_diff), dtype=torch.bool, device=device)
    diff_mask[:, -num_diff:] = True
    padded_ids = torch.nn.functional.pad(input_ids, (0, num_diff), value=model.tokenizer.pad_token_id)

    print(f"\nProfiling {sampling_timesteps} diffusion steps, batch={batch}, "
          f"prefix_len={input_ids.shape[1]}, device={device}")
    print(f"Sampler: {sampler}, CFG: {cls_free_guidance}\n")

    for step_idx, (t, t_next) in enumerate(tqdm(time_pairs, desc='profiling', total=sampling_timesteps)):
        with profiler.track("noise_schedule"):
            alpha2 = sample_noise_schedule(t).unsqueeze(-1)
            alpha2_next = sample_noise_schedule(t_next).unsqueeze(-1)

        # --- Soft prompt generation ---
        with profiler.track("soft_prompt_generator"):
            soft_prompt, time_emb = model.soft_prompt_generator(z_t, alpha2)

        # --- GPT-2 forward pass ---
        with profiler.track("lm_embedding"):
            input_embed = model.lm_embedding(padded_ids).float()

        input_embed[diff_mask] = rearrange(soft_prompt, 'b l d -> (b l) d')

        with profiler.track("gpt2_forward"):
            from star_ldm.models.transfusion import _get_lm_dtype
            gpt2_outputs = model.gpt2(
                inputs_embeds=input_embed.to(_get_lm_dtype()),
                labels=None,
                output_hidden_states=True,
            )

        with profiler.track("extract_diffusion_hidden"):
            diffusion_tokens = rearrange(
                gpt2_outputs.hidden_states[-1][diff_mask],
                '(b l) d -> b l d', b=batch, l=num_diff
            )

        # --- Score network ---
        with profiler.track("concat_for_scorenet"):
            diffusion_tokens = torch.cat((soft_prompt, diffusion_tokens), dim=-1)

        with profiler.track("score_net_head"):
            v_pred = model.score_net_head(diffusion_tokens, time_emb)

        # --- Convert v to x0 and eps ---
        with profiler.track("v_to_x0_eps"):
            from star_ldm.diffusion.diff_utils import predict_start_from_v, predict_noise_from_v
            x_start = predict_start_from_v(z_t, v_pred, alpha2)
            eps = predict_noise_from_v(z_t, v_pred, alpha2)

        # --- DDPM/DDIM step ---
        if t_next[0] <= 0:
            z_t = x_start
            continue

        with profiler.track("ddpm_step"):
            if sampler == 'ddim':
                z_t = x_start * alpha2_next.sqrt() + eps * (1 - alpha2_next).sqrt()
            elif sampler == 'ddpm':
                noise = torch.randn_like(z_t)
                alpha2_now = alpha2 / alpha2_next
                min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
                max_var = (1.0 - alpha2_now)
                sigma = torch.exp(var_lambda * torch.log(max_var) + (1 - var_lambda) * torch.log(min_var))
                z_t = 1 / alpha2_now.sqrt() * (z_t - (1 - alpha2_now) / (1 - alpha2).sqrt() * eps) + torch.sqrt(sigma) * noise

    # --- Final soft prompt + GPT-2 generation ---
    with profiler.track("final_soft_prompt"):
        alpha2_final = torch.full((batch, 1), 1 - sigma2, device=device)
        noised = variance_preserving_map(x_start, alpha2_final)
        soft_prompt, _ = model.soft_prompt_generator(noised, alpha2_final)

    with profiler.track("final_embed_prep"):
        input_embed = model.lm_embedding(input_ids).float()
        input_embed = torch.cat((input_embed, soft_prompt), dim=1)

    with profiler.track("gpt2_generate"):
        from star_ldm.models.transfusion import _get_lm_dtype
        gen_ids = model.gpt2.generate(
            inputs_embeds=input_embed.to(_get_lm_dtype()),
            do_sample=True, max_new_tokens=32,
            pad_token_id=model.tokenizer.eos_token_id,
            top_p=0.9, repetition_penalty=1.2,
        )
    generation = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return generation


# ---------------------------------------------------------------------------
# Dummy model for profiling without checkpoint
# ---------------------------------------------------------------------------

def create_dummy_model(device):
    """Create a STAR-LDM model with random weights for profiling structure."""
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
    model = model.to(device)
    model.eval()
    return model


def load_real_model(model_path, device):
    """Load from checkpoint."""
    from star_ldm.interface import TransfusionGPTInterface
    interface = TransfusionGPTInterface(model_path=model_path, device=str(device))
    return interface.model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Profile STAR-LDM inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to checkpoint directory')
    parser.add_argument('--dummy', action='store_true',
                        help='Use random weights (no checkpoint needed)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--prompt', type=str, default='The meaning of life is')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup diffusion steps before profiling')
    parser.add_argument('--trace', type=str, default=None,
                        help='Path to save Chrome trace JSON (CUDA only)')
    args = parser.parse_args()

    if not args.dummy and args.model_path is None:
        parser.error('Provide --model_path or --dummy')

    # Select device
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
    if device.type == 'mps':
        print(f"Apple Silicon MPS backend")
    elif device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    print("Loading model...")
    if args.dummy:
        model = create_dummy_model(device)
        print("Using dummy model (random weights)")
    else:
        model = load_real_model(args.model_path, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Tokenize prompt
    input_ids = model.tokenizer(args.prompt, return_tensors='pt').input_ids.to(device)
    print(f"Prompt: '{args.prompt}' ({input_ids.shape[1]} tokens)")

    # Warmup run (important for MPS which compiles kernels on first use)
    if args.warmup > 0:
        print(f"\nWarmup ({args.warmup} steps)...")
        warm_profiler = ManualProfiler(device)
        with torch.no_grad():
            profile_sampling(model, input_ids, warm_profiler,
                             sampling_timesteps=args.warmup)
        print("Warmup done.")

    # Actual profiling run
    print(f"\n{'='*80}")
    print(f"PROFILING RUN: {args.sampling_timesteps} diffusion steps")
    print(f"{'='*80}")

    profiler = ManualProfiler(device)

    # If CUDA, also use torch.profiler for trace
    use_torch_profiler = (device.type == 'cuda' and args.trace is not None)

    if use_torch_profiler:
        from torch.profiler import profile, ProfilerActivity, record_function
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as torch_prof:
            with torch.no_grad():
                generation = profile_sampling(
                    model, input_ids, profiler,
                    sampling_timesteps=args.sampling_timesteps)
        torch_prof.export_chrome_trace(args.trace)
        print(f"\nChrome trace saved to: {args.trace}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")
        print("\nTorch profiler top ops:")
        print(torch_prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    else:
        with torch.no_grad():
            generation = profile_sampling(
                model, input_ids, profiler,
                sampling_timesteps=args.sampling_timesteps)

    # Print results
    print(f"\nGenerated: {generation[:200]}...")
    summary = profiler.summary()

    # Compute per-step breakdown
    steps = args.sampling_timesteps
    print(f"\nPer-step average (over {steps} steps):")
    print(f"  GPT-2 forward:        {summary.get('gpt2_forward', {}).get('avg_ms', 0):.2f} ms")
    print(f"  Soft prompt gen:      {summary.get('soft_prompt_generator', {}).get('avg_ms', 0):.2f} ms")
    print(f"  Score net head:       {summary.get('score_net_head', {}).get('avg_ms', 0):.2f} ms")
    print(f"  DDPM step:            {summary.get('ddpm_step', {}).get('avg_ms', 0):.2f} ms")
    print(f"  Noise schedule:       {summary.get('noise_schedule', {}).get('avg_ms', 0):.2f} ms")
    print(f"  v→x0,eps conversion:  {summary.get('v_to_x0_eps', {}).get('avg_ms', 0):.2f} ms")

    total_diffusion = sum(v['total_ms'] for k, v in summary.items()
                          if k not in ('final_soft_prompt', 'final_embed_prep', 'gpt2_generate'))
    total_gen = summary.get('gpt2_generate', {}).get('total_ms', 0)
    print(f"\n  Diffusion loop total: {total_diffusion:.1f} ms")
    print(f"  AR generation total:  {total_gen:.1f} ms")
    print(f"  End-to-end:           {total_diffusion + total_gen:.1f} ms")


if __name__ == '__main__':
    main()
