# Poster Brief

Make me a 40x30 inch landscape academic research poster. Cornell CS class project. Clean, modern, top-conference quality. Cornell red (#B31B1B) accent.

## What this project is

I'm a co-author on STAR-LDM (COLM 2025), a language model that uses 50 steps of latent diffusion to plan what to say before generating text autoregressively with GPT-2 Large. It produces much better text (MAUVE 94.6 vs 85.2 for GPT-2 Large) but is slow because each diffusion step runs the full 770M-parameter GPT-2 backbone. My CS 5220 (parallel computing) project optimizes this inference pipeline using profiling, custom Metal GPU kernels for Apple Silicon, and structural code changes.

## The story

We profiled STAR-LDM and found that GPT-2 accounts for 77% of runtime. We then discovered that the bottleneck wasn't GPU compute but framework overhead: HuggingFace issues 6,185 operator dispatches per diffusion step, of which only 432 do actual math. We wrote 6 custom Metal GPU kernels (1,700 lines of code) but only 2 of 6 were faster than PyTorch's defaults. The biggest win came from rewriting the GPT-2 forward pass in 40 lines of plain PyTorch (3.6x faster) and adding KV-cache with pre-allocated buffers that make per-step cost O(1) instead of O(prefix_length). At 1000-token prefixes with 50 diffusion steps, this gives 9.46x end-to-end speedup.

## Poster layout guidance from professor

- ~25% introduction/background on the architecture  
- ~50% what we accomplished and results
- ~25% future work (what we'll do between poster session and final report)
- Less text is preferred. Figures and tables dominate.

## Background reading

The STAR-LDM paper: https://arxiv.org/abs/2602.20528

The roofline model (Williams et al. 2009) is how we characterized hardware utilization. FlashAttention (Dao et al. NeurIPS 2022) is the canonical example of GPU kernel fusion for transformers — we target the complementary regime where sequences are tiny (8 tokens) and tiling is overhead. DeepSpeed Inference (Aminabadi et al. SC 2022) identified kernel launch overhead as the bottleneck at small batch sizes, matching our finding.

## Architecture details

STAR-LDM (956M params total):
- Soft Prompt Generator: 6 transformer layers, dim 1024, 16 heads. Takes noised 768-dim sentence embedding, outputs 8 soft prompt tokens in 1280-dim GPT-2 space.
- GPT-2 Large: 36 layers, dim 1280, 20 heads, 770M params. Processes the 8 soft prompts with cached prefix KV.
- Score Network Head: 6 transformer layers, dim 1024. Takes concatenated (soft prompts, GPT-2 hidden states) = 2560-dim input, outputs 768-dim v-prediction.
- DDPM update produces z_{t-1} from z_t using the v-prediction. Repeat 50 times.

Quality results from the paper:
| Model | Params | MAUVE |
|---|---|---|
| GPT-2 Large | 770M | 85.2 |
| GPT-2 XL | 1.5B | 86.6 |
| STAR-LDM | 956M | 94.6 |

## All experimental data

### Baseline profiling (50 C4 validation prompts, unoptimized, 50 steps)

| Component | Mean ms | % of total | Calls per generation |
|---|---|---|---|
| GPT-2 forward (diffusion loop) | 884 | 38.7% | 50 |
| GPT-2 generate (AR decode) | 879 | 38.5% | 1 |
| SoftPromptGenerator | 208 | 9.1% | 51 |
| ScoreNetHead | 211 | 9.2% | 50 |
| DDPM step + noise schedule | 31 | 1.4% | 50 |
| Other | 70 | 3.1% | — |
| Total | 2,282 | 100% | |

### Dispatch count per v-prediction call

HuggingFace GPT-2: 6,185 ATen dispatches
Our streamlined forward: 432 ATen dispatches
Reduction: 93%

Top dispatch consumers in HuggingFace:
- scaled_dot_product_attention: 360 calls
- where/arange/le (mask construction): 806 calls  
- addmm (linear projections): 1,440 calls
- view/empty/copy_ (tensor management): 16,090 calls

### Roofline analysis (M4 Max: 14 TFLOPS FP32, 546 GB/s, ridge point 25.6 FLOP/byte)

| Operation | Time (μs) | Arith. Intensity | Achieved GFLOP/s | Ceiling GFLOP/s | Efficiency |
|---|---|---|---|---|---|
| RMSNorm+FiLM (JIT) | 27 | 0.74 | 2.1 | 402 | 0.5% |
| RMSNorm+FiLM (Metal) | 13 | 0.74 | 4.5 | 402 | 1.1% |
| Tiny Attention (JIT) | 95 | 2.28 | 3.2 | 1,245 | 0.3% |
| Tiny Attention (Metal) | 40 | 2.28 | 7.4 | 1,245 | 0.6% |
| DDPM Step (JIT) | 41 | 1.25 | 0.4 | 682 | 0.1% |
| GPT-2 decode-8 (KV cached) | 95,560 | 4.00 | 120 | 2,184 | 5.5% |
| SoftPromptGenerator (6 layers) | 2,775 | 4.04 | 440 | 2,206 | 19.9% |

Key finding: micro-transformer ops achieve <1% of roofline ceiling. Dispatch-latency-bound regime.

### Metal kernel microbenchmarks (correct model dimensions)

| Kernel | JIT (ms) | Metal (ms) | Speedup | Status |
|---|---|---|---|---|
| RMSNorm+FiLM (1×8×1024) | 0.027 | 0.016 | 1.69x | Used |
| Tiny Attention (1×16×8×64) | 0.099 | 0.042 | 2.39x | Used |
| DDPM Step (1×768) | 0.040 | 0.115 | 0.35x | Disabled |
| Decode-N Attn (S=13) | 0.021 | 0.011 | 1.97x | Not integrated |
| Fused FFN (8×1280→5120→1280) | 0.124 | 15.77 | 0.01x | Failed |
| Spec Verify (K=4, V=50257) | 0.939 | 7.214 | 0.13x | Not used |

6 kernels written (1,700 lines Metal + Obj-C++). 2 deliver speedups. 4 negative results.

### Prefix length scaling — THE MAIN RESULT

**50 Diffusion Steps (5 runs each, 95% CI):**

| Prefix | Baseline (ms) | Optimized (ms) | Speedup |
|---|---|---|---|
| 16 | 1,536 ± 4 | 1,154 ± 6 | 1.33x |
| 64 | 1,987 ± 7 | 1,172 ± 4 | 1.70x |
| 128 | 2,557 ± 10 | 1,197 ± 6 | 2.14x |
| 256 | 4,099 ± 73 | 1,356 ± 20 | 3.02x |
| 512 | 7,507 ± 134 | 1,569 ± 20 | 4.79x |
| 768 | 9,745 ± 170 | 1,623 ± 12 | 6.01x |
| 900 | 12,130 ± 272 | 1,717 ± 13 | 7.07x |
| 1000 | 14,377 ± 91 | 1,519 ± 9 | 9.46x |

**20 Diffusion Steps (5 runs each, 95% CI):**

| Prefix | Baseline (ms) | Optimized (ms) | Speedup |
|---|---|---|---|
| 16 | 858 ± 4 | 722 ± 9 | 1.19x |
| 64 | 1,053 ± 6 | 748 ± 2 | 1.41x |
| 128 | 1,296 ± 3 | 777 ± 6 | 1.67x |
| 256 | 1,801 ± 9 | 836 ± 5 | 2.15x |
| 512 | 3,001 ± 20 | 973 ± 4 | 3.09x |
| 768 | 4,255 ± 23 | 1,158 ± 7 | 3.68x |
| 900 | 5,180 ± 33 | 1,279 ± 7 | 4.05x |
| 1000 | 5,421 ± 10 | 995 ± 4 | 5.45x |

Baseline cost is O(prefix_length × steps). Optimized cost is approximately O(steps). The gap grows linearly.

### Batch size scaling (50 steps, fused + KV-cache)

| Batch | Per-sample (ms) | Throughput (samples/s) | Scaling |
|---|---|---|---|
| 1 | 1,339 | 0.75 | 1.00x |
| 2 | 963 | 1.04 | 1.39x |
| 4 | 747 | 1.34 | 1.79x |
| 8 | 737 | 1.36 | 1.82x |

### Negative results

- Picard parallel diffusion: convergence too slow. After 20 iterations, cosine similarity to sequential reference = 0.78. GPT-2's denoising function is too nonlinear for fixed-point iteration.
- torch.compile on MPS: both inductor and aot_eager make GPT-2 2-3x slower.
- Fused FFN Metal kernel: 127x slower than Apple BLAS. Cannot out-kernel the vendor's matrix multiply.
- Decode-N attention kernel: 1.97x faster at short KV (S<40), slower at longer lengths. Zero net improvement end-to-end because Python wrapper overhead consumes the per-call savings.

## Future work (for between poster session and final report)

1. Cross-platform CUDA comparison on NVIDIA A100
2. Quality validation: 5,000 C4 continuations, compute MAUVE + perplexity to verify optimizations preserve output quality
3. Model scheduling: skip GPT-2 at easy noise levels, use cheap SPG+ScoreNet only (~3ms vs ~12ms/step)
4. Consistency distillation: train student to go from 50 steps to 4 (stretch goal)

## Hardware

Apple M4 Max, 40-core GPU, 128 GB unified memory, 546 GB/s bandwidth. PyTorch 2.10, MPS backend. Metal Shading Language for custom GPU kernels.
