# STAR-LDM Optimization Plan

## Project: Inference & Training Optimization for Hybrid Diffusion-AR Language Models
**Course**: Cornell CS 5220 — Applications of Parallel Computers (Spring 2026)
**Target hardware**: Apple M4 Max (128GB unified memory) + NVIDIA GPU (for comparison)

---

## 1. Project Overview

STAR-LDM is a hybrid architecture that combines diffusion planning in a 768-dim Sentence-T5 embedding space with autoregressive generation via GPT-2 Large. During inference, a 50-step diffusion loop iteratively denoises a semantic embedding, converts it to 8 soft prompt tokens, and feeds those into GPT-2 to generate text.

This project optimizes STAR-LDM inference (and select training operations) through:
- Profiling to identify bottlenecks
- Custom fused kernels (Triton for CUDA, Metal for Apple Silicon)
- Architectural restructuring (KV-cache reuse)
- Cross-platform performance comparison (CUDA vs MPS/Metal vs CPU)

The work has three publishable angles:
- **Angle 1 (Systems)**: First systematic characterization + optimization of hybrid diffusion-AR inference
- **Angle 2 (Architecture)**: Cross-platform analysis showing unified memory changes optimization strategy
- **Angle 3 (ML)**: Distilling the 50-step diffusion to 1-4 steps (consistency/progressive distillation)

---

## 2. Architecture Summary

### Components (from `star_ldm/models/transfusion.py`)

| Component | Class | Params | Input → Output |
|-----------|-------|--------|----------------|
| AR Decoder | GPT-2 Large (HuggingFace) | 770M | `(B, seq, 1280)` embeddings → logits + hidden states |
| Soft Prompt Generator | `SoftPromptGenerator` | ~80M | `(B, 768)` noised emb + α² → `(B, 8, 1280)` soft prompts |
| Score Network | `ScoreNetHead` | ~80M | `(B, 8, 2560)` concat tokens + time_emb → `(B, 768)` v-prediction |
| Sentence Encoder | Sentence-T5 XL (frozen) | — | text → `(B, 768)` embedding |
| Classifier (optional) | `NoiseConditionedMLP` | ~15M | `(B, 768)` noised emb + α² → `(B, 1)` logits |

### Key dimensions
- Sentence embedding: 768
- LM embedding (GPT-2): 1280
- Transformer dim (prompt gen + score net): 1024
- Num heads: 16 (dim_head=64)
- Prompt length: 8 tokens
- Transformer depth: 6 layers (both prompt gen and score net)

### Inference flow (`TransfusionGPT.sample()`, line 230)
```
z_T ~ N(0, I)                              # (B, 768)
for step = 1..50:
    α² = noise_schedule(t)                  # scalar per batch
    soft_prompts = SoftPromptGenerator(z_t, α²)  # (B, 8, 1280)
    input_embed = LM_embedding(prefix_ids)       # (B, prefix_len, 1280)
    input_embed[diff_positions] = soft_prompts
    hidden = GPT2(input_embed)                    # FULL forward pass
    diff_hidden = hidden[diff_positions]           # (B, 8, 1280)
    v = ScoreNetHead(cat(soft_prompts, diff_hidden), time_emb)  # (B, 768)
    x₀, ε = v_to_predictions(z_t, v, α²)
    z_{t-1} = ddpm_step(z_t, x₀, ε, α², α²_next)
final_soft_prompts = SoftPromptGenerator(z_0, α²=0.95)
generation = GPT2.generate(prefix_embed + final_soft_prompts)
```

---

## 3. Profiling Plan

### 3.1 Manual profiler (DONE — `scripts/profile_inference.py`)
- Uses `torch.mps.synchronize()` / `torch.cuda.synchronize()` + `time.perf_counter()`
- Wraps each component: `gpt2_forward`, `soft_prompt_generator`, `score_net_head`, `ddpm_step`, `noise_schedule`, `v_to_x0_eps`, `gpt2_generate`
- Outputs per-component timing table with call counts and percentages
- Run: `python -m scripts.profile_inference --dummy --sampling_timesteps 50`
- Run with real checkpoint: `python -m scripts.profile_inference --model_path checkpoints/star-ldm`

### 3.2 Xcode Metal System Trace (for Apple Silicon)
```bash
xcrun xctrace record --template 'Metal System Trace' --launch -- \
    .venv/bin/python -m scripts.profile_inference --dummy --sampling_timesteps 10
```
Shows: GPU utilization, kernel dispatch timeline, memory bandwidth, command buffer gaps.

### 3.3 NVIDIA Nsight Systems (for CUDA comparison)
```bash
nsys profile -o star_ldm_profile python -m scripts.profile_inference --model_path checkpoints/star-ldm
```

### 3.4 torch.profiler with Chrome trace (CUDA only)
```bash
python -m scripts.profile_inference --model_path checkpoints/star-ldm --trace trace.json
# Open trace.json in chrome://tracing or https://ui.perfetto.dev/
```

### Expected bottleneck ranking (to be confirmed by profiling)

| Rank | Component | Calls per generation | Type | Est. % time |
|------|-----------|---------------------|------|-------------|
| 1 | GPT-2 full forward | 50x | Compute-bound | 50-70% |
| 2 | GPT-2 autoregressive generate | 1x | Sequential/compute | 15-30% |
| 3 | SoftPromptGenerator (6-layer transformer, 8 tokens) | 50x | Kernel-launch overhead | 5-10% |
| 4 | ScoreNetHead (6-layer transformer, 8 tokens) | 50x | Kernel-launch overhead | 3-5% |
| 5 | DDPM step arithmetic | 49x | Kernel-launch overhead (tiny tensors) | 2-5% |
| 6 | v→x₀,ε conversion | 50x | Elementwise | <1% |
| 7 | Noise schedule evaluation | 50x | Elementwise | <1% |

---

## 4. Optimization Implementations

### 4.1 KV-Cache Reuse for GPT-2 (HIGHEST IMPACT)

**File**: `star_ldm/models/transfusion.py` — new method `sample_with_kv_cache()`

**Problem**: GPT-2 runs a full forward pass (36 layers, entire sequence) at every diffusion step, but only 8 out of `prefix_len + 8` tokens change between steps. The prefix tokens produce identical K,V tensors every time.

**Implementation**:
```python
def sample_with_kv_cache(self, input_ids, ...):
    # Step 1: Compute KV cache for prefix (ONE TIME)
    prefix_embed = self.lm_embedding(input_ids)
    prefix_out = self.gpt2(inputs_embeds=prefix_embed, use_cache=True, output_hidden_states=True)
    cached_kv = prefix_out.past_key_values  # 36 layers × (K, V)

    # Step 2: Diffusion loop — only process 8 new tokens
    for step in range(sampling_timesteps):
        soft_prompt = self.soft_prompt_generator(z_t, alpha2)  # (B, 8, 1280)
        gpt2_out = self.gpt2(
            inputs_embeds=soft_prompt,
            past_key_values=cached_kv,  # Reuse prefix KV!
            use_cache=False,
            output_hidden_states=True,
        )
        # ... rest of denoising step
```

**Speedup**: For 64-token prefix, GPT-2 processes 72 tokens → 8 tokens per step. ~8x on GPT-2 component, ~4-6x overall.

**Validation**: Compare generation output with and without KV-cache to ensure identical results (within floating point tolerance).

**Complexity**: Medium. HuggingFace GPT-2 already supports `past_key_values`. Main work is restructuring the `sample()` method and verifying hidden states are extracted correctly.

### 4.2 Fused RMSNorm + FiLM Kernel

**Files to modify**: `star_ldm/models/modules/blocks.py` (Attention.forward, FeedForward.forward)

**Current code** (3 kernel launches, `blocks.py:124-131`):
```python
x = self.pre_norm(x)                          # Kernel 1: RMSNorm
scale, shift = self.time_cond(time_emb).chunk(2, dim=-1)  # Kernel 2: SiLU + Linear
x = (x * (scale + 1)) + shift                 # Kernel 3: FiLM modulation
```

**Current RMSNorm** (`norm.py:5-12`):
```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma
```

**Fused Triton kernel** (`kernels/triton/fused_rmsnorm_film.py`):
```python
@triton.jit
def fused_rmsnorm_film_kernel(X, GAMMA, SCALE_SHIFT, OUT, stride, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Phase 1: RMSNorm — compute ||x|| and normalize
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(X + row * stride + cols, mask=mask).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x) / D + 1e-8)
    x_normed = x / norm * tl.sqrt(float(D))  # self.scale = dim**0.5
    gamma = tl.load(GAMMA + cols, mask=mask).to(tl.float32)
    x_normed = x_normed * gamma

    # Phase 2: FiLM — apply scale and shift (precomputed, passed in)
    film_scale = tl.load(SCALE_SHIFT + (row // 8) * 2 * D + cols, mask=mask).to(tl.float32)  # broadcast across seq
    film_shift = tl.load(SCALE_SHIFT + (row // 8) * 2 * D + D + cols, mask=mask).to(tl.float32)
    out = x_normed * (film_scale + 1.0) + film_shift

    tl.store(OUT + row * stride + cols, out.to(tl.float16), mask=mask)
```

**Fused Metal kernel** (`kernels/metal/fused_rmsnorm_film.metal`):
```metal
kernel void fused_rmsnorm_film(
    device const float *x [[buffer(0)]],        // (B*8, 1024)
    device const float *gamma [[buffer(1)]],     // (1024,)
    device const float *scale_shift [[buffer(2)]],// (B, 2*1024) [scale|shift]
    device float *out [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])        // (row, col_block)
{
    // Phase 1: RMSNorm across dim in threadgroup
    // Phase 2: Apply FiLM scale/shift
}
```

**Where used**: Both `Attention.forward()` and `FeedForward.forward()` in the SoftPromptGenerator and ScoreNetHead. That's 12 layers × 2 = 24 calls per diffusion step × 50 steps = **1200 invocations** reduced from 3600 kernel launches to 1200.

### 4.3 Fused 8-Token Attention (No-Tile Attention)

**File**: New kernel, called from `blocks.py:154` replacing `F.scaled_dot_product_attention`

**Problem**: Standard SDPA/FlashAttention is designed for long sequences. For 8 tokens with 16 heads and dim_head=64, the entire attention matrix is 8×8=64 values per head — fits entirely in registers. No tiling needed.

**Triton kernel** (`kernels/triton/tiny_attention.py`):
```python
@triton.jit
def tiny_attention_kernel(
    Q, K, V, OUT,
    stride_qb, stride_qh, stride_qs, stride_qd,
    scale,
    SEQ_LEN: tl.constexpr,  # 8
    HEAD_DIM: tl.constexpr,  # 64
):
    batch_head = tl.program_id(0)  # one program per (batch, head)
    # Load entire Q, K, V for this head (8×64 each) into registers
    # Compute 8×8 QK^T → full softmax (no online algorithm) → 8×64 output
    # Single kernel, zero shared memory, zero tiling
```

**Also fuses QK-norm**: The current code does separate RMSNorm on Q and K (`blocks.py:135`). The fused kernel computes QK-norm inline.

**Where used**: 12 attention layers × 50 steps = 600 calls. Currently each is a separate SDPA dispatch.

### 4.4 Fused DDPM Denoising Step

**File**: New kernel, called from `transfusion.py:276-282`

**Current code** (~17 separate kernel launches per step on (B, 768) tensors):
```python
noise = torch.randn_like(z_t)
alpha2_now = alpha2/alpha2_next
min_var = torch.exp(torch.log1p(-alpha2_next) - torch.log1p(-alpha2)) * (1.0 - alpha2_now)
max_var = (1.0 - alpha2_now)
sigma = torch.exp(var_lambda * torch.log(max_var) + (1 - var_lambda) * torch.log(min_var))
z_t = 1/alpha2_now.sqrt() * (z_t - (1-alpha2_now)/(1-alpha2).sqrt() * eps) + torch.sqrt(sigma) * noise
```

**Fused kernel**: One kernel that reads z_t, eps, noise, alpha2, alpha2_next and writes z_{t-1}. All scalar arithmetic (the variance interpolation) done once in registers, then broadcast across 768 dims.

**Triton** (`kernels/triton/fused_ddpm_step.py`): ~80 lines
**Metal** (`kernels/metal/fused_ddpm_step.metal`): ~50 lines
**CPU fallback** (for Apple Silicon unified memory): plain C++ with OpenMP, zero dispatch overhead

### 4.5 Fused RMSNorm + FiLM + GLU FFN

**File**: New kernel replacing `FeedForward.forward()` entirely

**Current ops** (blocks.py:79-88):
1. RMSNorm(x) — kernel 1
2. time_cond(time_emb) → scale, shift — kernel 2 (SiLU + Linear)
3. FiLM: x * (scale+1) + shift — kernel 3
4. GLU: Linear(x) → split → SiLU(gate) * val — kernel 4
5. Dropout — kernel 5
6. Linear(glu_out) → output — kernel 6

**Fused**: Kernels 1-3 fused (RMSNorm-FiLM, see 4.2). Kernel 4 can be fused with 3 using CUTLASS epilogue fusion pattern (RMSNorm-FiLM output feeds directly into GLU GEMM without global memory round-trip).

For 8-token sequences, the GLU GEMM is `(8, 1024) × (1024, 1364)` — tiny. On GPU this is memory-bound, so fusing the read of x with the GEMM input avoids one global memory read.

### 4.6 CFG Batching

**File**: `star_ldm/models/transfusion.py`, `diffusion_model_predictions()` (line 420)

**Current**: When `cls_free_guidance != 1.0`, calls `v_pred()` twice per step (conditional + unconditional).

**Fix**: Batch them:
```python
if cls_free_guidance != 1.0:
    # Stack conditional and unconditional inputs into batch dim
    z_t_double = torch.cat([z_t, z_t], dim=0)
    alpha2_double = torch.cat([alpha2, alpha2], dim=0)
    ids_double = torch.cat([input_ids, input_ids], dim=0)
    mask_double = torch.cat([diffusion_token_mask, diffusion_token_mask], dim=0)

    # Single forward with drop_cond flags
    # ... custom logic to apply null prompt to second half of batch
    # Halves GPT-2 forward passes from 100 to 50
```

### 4.7 Training Kernels

#### Fused v-target + MSE + Weighting (`kernels/triton/fused_v_loss.py`)
Replaces `transfusion.py:393-409`. Computes:
- v_target = √α² · ε - √(1-α²) · x₀
- MSE = mean((v_pred - v_target)²) per sample
- weight = lookup_table[γ]
- weighted_loss = weight × MSE

All in one kernel: reads v_pred(768), x₀(768), ε(768), α²(1), γ(1). Writes weighted_loss(1), unweighted_loss(1).

#### Fused Variance-Preserving Noising (`kernels/triton/fused_vp_map.py`)
Replaces `variance_preserving_map()`. One kernel: z_t = √α² · x + √(1-α²) · ε.

#### Fused EMA Bin Update (`kernels/triton/fused_ema_update.py`)
Replaces `time_sampler.py:74-94`. Parallelizes across 100 bins. Each bin accumulates losses from the batch, then applies EMA update.

---

## 5. Cross-Platform Strategy (Angle 2)

### Kernel implementations per platform

| Kernel | Triton (CUDA) | Metal (Apple) | CPU (Apple unified) |
|--------|---------------|---------------|---------------------|
| Fused RMSNorm-FiLM | ✓ | ✓ | — |
| Tiny 8-token attention | ✓ | ✓ | — |
| Fused RMSNorm-FiLM-GLU | ✓ | ✓ | — |
| Fused DDPM step | ✓ | ✓ | ✓ (best on Apple!) |
| Fused v-loss | ✓ | ✓ | — |
| Fused VP noising | ✓ | ✓ | — |
| Fused EMA update | ✓ | — | — |
| KV-cache reuse | PyTorch | PyTorch | — |
| CFG batching | PyTorch | PyTorch | — |

### The unified memory insight

On Apple Silicon, the DDPM step kernel is better run on **CPU** because:
1. No kernel dispatch overhead (~5-10μs saved per step)
2. No memory transfer (CPU and GPU share memory)
3. The data is tiny: 3 × 768 floats = 9KB — fits in L1 cache
4. CPU can pipeline this while GPU processes the next soft prompt generator

This heterogeneous CPU/GPU split is **impossible on discrete GPU** (PCIe transfer cost > compute cost).

### Experiments for the paper

1. **Baseline profiling**: Same model, same inputs, CUDA vs MPS vs CPU — component-level timing
2. **Per-optimization impact**: Enable each optimization one-by-one, measure on both platforms
3. **Prefix length sweep**: Vary prefix from 16 to 512 tokens — shows KV-cache memory pressure
4. **Batch size sweep**: Vary B from 1 to 16 — shows compute utilization differences
5. **Kernel-level traces**: Xcode Metal System Trace vs Nsight Systems side-by-side

---

## 6. Distillation Plan (Angle 3)

### Phase 1: Characterize baseline (2 weeks)

```python
# Sweep step counts without distillation
for n_steps in [1, 2, 4, 8, 16, 25, 50, 100, 250]:
    generations = model.sample(prompts, sampling_timesteps=n_steps)
    # Evaluate: perplexity, MAUVE, LLM-as-judge coherence
```

Also measure ODE trajectory curvature:
- Run 250-step DDIM, record all intermediate z_t
- Compute deviation from straight line z_T → x_0
- If nearly straight → rectified flow will work well
- If curved → consistency distillation is better

### Phase 2: Implement distillation approaches (4 weeks)

#### Approach A: Consistency Distillation (Song et al., 2023)
Train f_θ(z_t, t) to map any point on the ODE trajectory to x₀.
Consistency loss: f_θ(z_t, t) ≈ f_θ(z_{t-Δ}, t-Δ).
Student architecture: same SoftPromptGenerator + ScoreNetHead, retrained.

#### Approach B: Progressive Distillation (Salimans & Ho, 2022)
Iteratively halve steps: 50→25→12→6→3→1.
Each round: teacher takes 2 DDIM steps, student takes 1 step to match.
Loss: ||z_{t-2}^teacher - z_{t-2}^student||².

#### Approach C: Rectified Flow (Liu et al., 2023)
Collect (z_T, x_0) pairs from pretrained model.
Train flow model: given z_t = (1-t)·x₀ + t·z_T, predict v = z_T - x₀.
Linear interpolation makes trajectories straight → 1-step Euler is exact.

### Phase 3: Evaluate (2 weeks)

| Method | Steps | Perplexity | MAUVE | LLM-judge | Embedding L2 |
|--------|-------|------------|-------|-----------|---------------|
| Teacher (STAR-LDM) | 50 | baseline | baseline | baseline | 0 |
| Naive truncation | 4 | ? | ? | ? | ? |
| Progressive distill | 4 | ? | ? | ? | ? |
| Consistency distill | 1 | ? | ? | ? | ? |
| Rectified flow | 1 | ? | ? | ? | ? |

---

## 7. File Structure (new files to create)

```
STAR-LDM/
├── kernels/
│   ├── __init__.py
│   ├── triton/
│   │   ├── __init__.py
│   │   ├── fused_rmsnorm_film.py      # Fused RMSNorm + FiLM conditioning
│   │   ├── tiny_attention.py           # 8-token no-tile attention
│   │   ├── fused_ddpm_step.py          # Fused DDPM denoising step
│   │   ├── fused_v_loss.py             # Fused training loss
│   │   ├── fused_vp_map.py             # Fused variance-preserving noising
│   │   └── fused_ema_update.py         # Fused EMA bin update
│   └── metal/
│       ├── fused_rmsnorm_film.metal    # Metal shader
│       ├── fused_rmsnorm_film.mm       # ObjC++ dispatch
│       ├── tiny_attention.metal
│       ├── tiny_attention.mm
│       ├── fused_ddpm_step.metal
│       ├── fused_ddpm_step.mm
│       └── setup.py                    # Build extension
├── scripts/
│   ├── profile_inference.py            # DONE — profiling script
│   ├── profile_training.py             # Training profiler (TODO)
│   ├── benchmark_kernels.py            # Microbenchmark individual kernels
│   └── evaluate_distillation.py        # Distillation evaluation
├── star_ldm/
│   └── models/
│       └── transfusion.py              # Modified: sample_with_kv_cache(), MPS compat
└── plan.md                             # This file
```

---

## 8. Implementation Order

### Phase 1: Profiling & Low-Hanging Fruit (Week 1-2)
1. ✅ MPS compatibility patches
2. ✅ Profiling script
3. ✅ Python environment setup
4. Download checkpoint and profile with real weights
5. Xcode Metal System Trace capture
6. Implement KV-cache reuse (biggest single speedup, pure PyTorch, no kernels)
7. Implement CFG batching (trivial, pure PyTorch)

### Phase 2: Fused Kernels (Week 3-5)
8. Fused DDPM step (simplest kernel — good first Triton/Metal exercise)
9. Fused RMSNorm + FiLM (medium complexity, used everywhere)
10. Fused 8-token attention (requires understanding attention mechanics)
11. Fused RMSNorm-FiLM-GLU (most complex fusion)
12. Microbenchmark each kernel vs baseline

### Phase 3: Training Kernels (Week 5-6)
13. Fused v-target + MSE + weighting
14. Fused variance-preserving noising
15. Fused EMA bin update
16. Profile training loop end-to-end

### Phase 4: Cross-Platform Analysis (Week 6-7)
17. Port Triton kernels to Metal
18. Implement CPU fallback for DDPM step
19. Run full benchmark suite on both platforms
20. Prefix length and batch size sweeps

### Phase 5: Distillation (Week 7-10)
21. Baseline step-count sweep
22. Trajectory curvature analysis
23. Implement progressive distillation
24. Implement consistency distillation
25. Implement rectified flow
26. Evaluation matrix

### Phase 6: Paper Writing (Week 10-12)
27. Systems paper (Angles 1+2): profiling + kernels + cross-platform
28. ML paper (Angle 3): distillation results
29. Figures: timeline traces, speedup curves, quality vs speed tradeoffs

---

## 9. Checkpoint Download

The pretrained STAR-LDM checkpoint is hosted on Cornell Box:
- **Model**: https://cornell.box.com/s/09kp1l61cmnejixpywqvg5vauoq8sih1
- **Sentiment classifier**: https://cornell.box.com/s/gukku7f1k14vjteiqjrqz7y033ept58w

Download and extract to `checkpoints/star-ldm/` (should contain `model.pt` + `args.yaml`).

---

## 10. Deeper Kernel Opportunities (Informed by Architecture Analysis)

The paper's key architectural insight — that the same 8 soft prompt tokens are processed through both a micro-transformer AND the full GPT-2 backbone every diffusion step — reveals three optimization opportunities at different levels.

### 10.1 GPT-2 Decode Attention on 8 Tokens (36 layers) — THE Dominant Cost

**Status**: Not implemented. This is 68% of each diffusion step and the single biggest optimization target.

With KV-cache, each of GPT-2's 36 attention layers performs:
```
Q: (1, 20 heads, 8, 64)          ← 8 soft prompt query tokens
K: (1, 20 heads, prefix+8, 64)   ← cached prefix KV + 8 new
V: (1, 20 heads, prefix+8, 64)
```

This is a **short-query, moderate-KV** attention pattern. Standard SDPA/FlashAttention is optimized for long sequences on both sides — the tiling, online softmax, and memory management machinery is pure overhead for 8 query tokens. This pattern executes **36 layers × 50 steps = 1800 times** per generation.

**Kernel opportunity: Fused "decode-8" attention**
- Keep all 8 query vectors in registers throughout
- Stream through the prefix KV cache once per layer
- No tiling needed on the query side (8 tokens fit trivially)
- Could batch attention across multiple layers to amortize KV cache reads

**Challenge**: This requires replacing PyTorch's internal attention implementation inside the HuggingFace GPT-2 model, which is significantly harder than wrapping an external function. Would need to hook into or replace `GPT2Attention.forward()`.

**Expected impact**: If we can reduce attention overhead by 2x on this pattern, that's ~34% of total step time → ~17% end-to-end speedup on the diffusion loop. Stacks multiplicatively with step reduction (consistency distillation).

### 10.2 Micro-Transformer Layer-Level Fusion (SPG + ScoreNetHead)

**Status**: Individual ops fused (RMSNorm+FiLM: 4.61x, Tiny Attention: 1.87x), but not fused across layers.

The SoftPromptGenerator has 6 transformer layers, each doing:
```
Per layer: Norm → Attention → Norm → FFN → FiLM modulate
         = ~5 kernel launches × 6 layers = 30 launches per SPG forward
```

For 8 tokens × 1024 dim, each kernel does almost no compute but pays full MPS dispatch overhead (~0.01-0.05ms each). 30 launches × 0.03ms ≈ **1ms of pure dispatch overhead** per SPG call.

**Kernel opportunity: Mega-kernel for entire transformer layer (or multi-layer)**
- Fuse Norm + Attention + Norm + FFN + FiLM into a single Metal dispatch
- For seq_len=8, the entire computation fits in threadgroup shared memory
- Could even fuse 2-3 layers into one dispatch since intermediate tensors are tiny (8 × 1024 = 32KB)
- The entire SPG (6 layers) might fit in a single mega-kernel dispatch

**Expected impact**: Eliminates ~30 kernel launches → ~1ms saved per step × 50 steps = ~50ms. Modest absolute savings but demonstrates the "kernel launch overhead dominates for micro-workloads" insight, which is a generalizable systems finding for any architecture using small conditioning transformers.

### 10.3 Inter-Sample Parallelism Across Batch Elements

**Status**: Not implemented or discussed.

When batch_size > 1, each sample's diffusion trajectory is completely independent. The noise, timestep schedule, and denoising path for sample A have no data dependency on sample B. This opens up:

**Opportunity A: Pipelined multi-sample execution**
With consistency distillation reducing steps to 4, pipeline multiple samples:
```
Timeline (sequential):
  [Sample A step 1] [Sample A step 2] [Sample A step 3] [Sample A step 4]
  [Sample B step 1] [Sample B step 2] [Sample B step 3] [Sample B step 4]

Timeline (pipelined, with CPU/GPU overlap):
  GPU: [A step 1] [A step 2 | B step 1 noise prep] [A step 3 | B step 2] ...
  CPU: [B noise prep] [A noise prep] ...
```

**Opportunity B: Different step counts per sample**
With consistency distillation, some samples may converge faster than others. An adaptive scheduler could give "easy" prompts 2 steps and "hard" prompts 4 steps, improving throughput.

**Applicability**: Primarily for throughput-oriented serving (batch > 1), not single-sample interactive generation. More relevant if STAR-LDM is deployed as a service.

### 10.4 Combined Impact Analysis

If all three levels are optimized:

| Optimization | Target | Per-step savings | End-to-end (50 steps) |
|---|---|---|---|
| GPT-2 decode-8 attention | 68% of step | ~7ms/step | ~350ms |
| Micro-transformer mega-kernel | 14% of step | ~1ms/step | ~50ms |
| Existing fused ops (DDPM, etc.) | 3% of step | ~0.3ms/step | ~15ms |
| **Total kernel savings** | | | **~415ms (28%)** |

Combined with consistency distillation (50→4 steps), the kernel savings become:
- 4 steps × 15ms/step (optimized) = **60ms** diffusion, vs 4 × 22ms = 88ms unoptimized
- The kernel work matters MORE with fewer steps because per-step overhead is a larger fraction

---

## 11. Speculative Decoding: Draft Model Distillation

### 11.1 Architecture

The AR generation step (~400ms, 27% of total) can be accelerated via speculative decoding with a distilled draft model.

**Key insight**: The draft model only needs to approximate the *marginal* next-token distribution of the full STAR-LDM pipeline given the prefix. It doesn't need to understand diffusion or soft prompts — just mimic the final token distribution.

**Draft model** (~20M params):
- 4 transformer decoder layers, 256 hidden dim, 4 attention heads, 1024 FFN dim
- Same GPT-2 tokenizer (50,257 vocab)
- ~26x smaller than GPT-2 Large backbone

**Training**:
1. Run full STAR-LDM on 50k-200k prefixes, collect (prefix → generated_tokens) pairs
2. Train with cross-entropy on completions (sequence-level knowledge distillation)
3. Literature suggests even 2k-4k samples can yield >80% acceptance for narrow domains

**Verification**: Draft generates K=4-8 tokens (fast), full pipeline verifies all K in one GPT-2 Large forward pass. Speculative decoding guarantees **exact sampling** from the target distribution — zero quality regression by construction.

**Expected impact**: 50-70% acceptance rate → AR step 400ms → ~270ms → ~130ms saved (8% of total)

### 11.2 Key References for Draft Distillation
- DistillSpec (Zhou et al., 2023) — foundational paper on draft distillation
- FastDraft (Intel, 2024) — 50-150M draft models, 3-stage training
- Scaling Laws for Speculative Decoding (2025) — acceptance_rate ∝ log(num_layers)
- LK Losses (2026) — direct acceptance rate optimization, 3-10% gains

### 11.3 Current Implementation Status

Speculative decoding infrastructure is implemented and working:
- `star_ldm/decoding/speculative.py` — custom loop supporting `inputs_embeds`
- `star_ldm/kernels/spec_verify.metal` — fused Metal verification kernel
- Handles `DynamicCache` from transformers 4.x+ (`.crop()` method)
- Handles different embedding dimensions between draft and target

Current limitation: Using GPT-2 XL as target model (wrong — target should be the STAR-LDM pipeline itself). Need to either:
- Distill a tiny draft model that approximates the full pipeline
- Or use the backbone GPT-2 Large as both draft and target (same model spec decode — only works if acceptance rate is high, which requires matched distributions)

---

## 12. Publishability Analysis

### 12.1 What's Needed for NeurIPS-Level

| Requirement | Status | Gap |
|---|---|---|
| Big headline number (≥3x speedup) | ❌ Currently 1.03-1.11x | Need consistency distillation |
| Quality preservation proof | ❌ Not evaluated | Need perplexity, MAUVE, human eval |
| Generalizable insight | ⚠️ Kernels are STAR-LDM specific | Need to frame as general diffusion-AR pattern |
| Rigorous ablation | ⚠️ Partial | Need per-component contribution at each step count |
| Comparison to baselines | ❌ No other diffusion-LM optimization work compared | Need to compare naive truncation, DDIM, etc. |

### 12.2 The Paper Story

**"Accelerating Inference in Diffusion-Augmented Language Models"**

Core thesis: Models that combine continuous diffusion planning with discrete AR generation have unique inference bottlenecks that neither diffusion speedups nor LLM speedups address alone. We present a unified optimization framework:

1. **Consistency distillation** for the diffusion planning stage (50→4 steps, ~2.5x on total)
2. **Hardware-aware micro-kernels** for the diffusion-AR interface operations (fixed-size attention, fused conditioning normalization, coupled transcendental updates)
3. **Distilled draft speculative decoding** for the AR generation stage (exact sampling, ~1.1x on total)
4. **KV-cache amortization** across the diffusion-AR boundary

Combined: **~3-4x end-to-end speedup** with provable (speculative decoding) or empirically validated (consistency distillation) quality preservation.

### 12.3 Comparable Published Works

| Paper | Venue | Technique | Speedup | Generality |
|---|---|---|---|---|
| FlashAttention | NeurIPS 2022 | IO-aware tiling | 2-4x | Any transformer |
| Speculative Decoding | ICML 2023 | Draft+verify | 2-3x | Any AR model |
| Medusa | ICML 2024 | Multi-head drafting | 2.2-3.6x | Any AR model |
| EAGLE | ICML 2024 | Feature-fusion drafting | 2-6x | Any AR model |
| LCM | 2023 | Consistency in latent space | 10-50x on diffusion | Any latent diffusion |
| **This work** | Target: NeurIPS 2026 | Unified diffusion-AR optimization | 3-4x total | Hybrid diffusion-AR models |

### 12.4 Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Consistency distillation doesn't converge | Medium | Progressive distillation as fallback (simpler, more stable) |
| Quality degrades at 4 steps | Low-Medium | 768-dim space is forgiving; can fall back to 8 steps (still 6x) |
| Draft model acceptance rate too low | Medium | Need sufficient training data; can use on-policy distillation |
| "Too narrow" reviewer rejection | High | Must frame as general framework, ideally show on ≥2 models |
| Timeline too tight for NeurIPS 2026 | High | EMNLP 2026 or NeurIPS workshop as backup venue |

---

## 13. Key References

### Diffusion Acceleration
- [Consistency Models (Song et al., ICML 2023)](https://arxiv.org/abs/2303.01469)
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — CD in continuous latent space, most relevant to STAR-LDM
- [Progressive Distillation (Salimans & Ho, ICLR 2022)](https://arxiv.org/abs/2202.00512)
- [Rectified Flow (Liu et al., 2023)](https://arxiv.org/abs/2209.03003)
- [Consistency Models Made Easy (ICLR 2025)](https://arxiv.org/abs/2406.14548)
- [Simplified Continuous-Time CMs (OpenAI, 2024)](https://arxiv.org/abs/2410.11081)
- [CDLM: Consistency Diffusion Language Models (2025)](https://arxiv.org/abs/2511.19269) — 14.5x speedup on discrete diffusion LMs
- [CD4LM (2026)](https://arxiv.org/abs/2601.02236) — 5.18x speedup, quality preserved

### Speculative Decoding & Draft Distillation
- [Speculative Decoding (Leviathan et al., ICML 2023)](https://arxiv.org/abs/2211.17192)
- [DistillSpec (Zhou et al., 2023)](https://arxiv.org/abs/2310.08461) — distillation for speculative decoding
- [FastDraft (Intel, 2024)](https://arxiv.org/abs/2411.11055) — 50-150M standalone draft models
- [Scaling Laws for Speculative Decoding (2025)](https://arxiv.org/abs/2505.07858)
- [LK Losses (2026)](https://arxiv.org/abs/2602.23881) — direct acceptance rate optimization
- [Online Speculative Decoding (ICML 2024)](https://arxiv.org/abs/2310.07177)
- [Medusa (ICML 2024)](https://arxiv.org/abs/2401.10774) — multi-head drafting
- [EAGLE (ICML 2024)](https://github.com/SafeAILab/EAGLE) — feature-fusion drafting
- [ReDrafter (Apple, 2024)](https://arxiv.org/abs/2403.09919) — RNN-based drafter, 2.3x on Apple Silicon
- [Training Domain Draft Models (ICLR 2025 SCOPE)](https://arxiv.org/abs/2503.07807) — best practices

### Systems & Kernels
- [FlashAttention (Dao et al., NeurIPS 2022)](https://arxiv.org/abs/2205.14135)
- [FlashFormer: Whole-Model Kernels](https://arxiv.org/html/2505.22758v1)
- [Deep Kernel Fusion for Transformers](https://arxiv.org/html/2602.11808)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Liger-Kernel (fused RMSNorm)](https://github.com/linkedin/Liger-Kernel)
- [Custom Metal Kernels for PyTorch](https://medium.com/@praburam_93885/custom-pytorch-operations-for-metal-backend-889736c6bc2a)

### STAR-LDM
- [STAR-LDM (Lovelace et al., COLM 2025)](https://arxiv.org/abs/2602.20528)
- [Latent Diffusion for Language Generation (NeurIPS 2023)](https://arxiv.org/abs/2212.09462)
