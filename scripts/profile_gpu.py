"""Minimal profiling script for Xcode Metal System Trace.

Run with:
    xcrun xctrace record --template 'Metal System Trace' \
        --output star_ldm_trace.trace \
        --time-limit 5s \
        --launch -- .venv/bin/python -m scripts.profile_gpu

Then open star_ldm_trace.trace in Instruments.app
"""
import torch
import time
import os

# Force MPS
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

device = 'mps'
from star_ldm.interface import TransfusionGPTInterface
from star_ldm.models.modules.fused_blocks import swap_to_fused_blocks

print("Loading model...")
interface = TransfusionGPTInterface(model_path='checkpoints/star-ldm', device=device)
model = interface.model
swap_to_fused_blocks(model.soft_prompt_generator.transformer)
swap_to_fused_blocks(model.score_net_head.transformer)

tokenizer = model.tokenizer
input_ids = tokenizer('The meaning of life is', return_tensors='pt').input_ids.to(device)

# Warmup
print("Warming up...")
with torch.no_grad():
    cached_kv = model._compute_prefix_kv_cache(input_ids)
    z_t = torch.randn(1, 768, device=device)
    from star_ldm.diffusion.noise_schedule import get_scaled_noise_schedule
    schedule = get_scaled_noise_schedule('cosine', scale=3.0)
    alpha2 = schedule(torch.tensor([0.5], device=device)).unsqueeze(-1)
    for _ in range(3):
        model._v_pred_cached(z_t, alpha2, cached_kv)
        cached_kv = model._compute_prefix_kv_cache(input_ids)
    torch.mps.synchronize()

# Profile region: 5 v_pred calls
print("PROFILING START")
torch.mps.synchronize()
time.sleep(0.1)  # small gap for trace alignment

with torch.no_grad():
    cached_kv = model._compute_prefix_kv_cache(input_ids)
    torch.mps.synchronize()

    for i in range(5):
        model._v_pred_cached(z_t, alpha2, cached_kv)
        torch.mps.synchronize()
        cached_kv = model._compute_prefix_kv_cache(input_ids)

torch.mps.synchronize()
print("PROFILING END")
print("Done. Open the .trace file in Instruments.app")
