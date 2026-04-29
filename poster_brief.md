# Poster Brief

Make me a 40x30 inch landscape academic research poster. Cornell CS class project. Clean, modern, top-conference quality. Cornell red (#B31B1B) accent.

## What this project is

I'm a co-author on STAR-LDM (COLM 2025), a language model that uses 50 steps of latent diffusion to plan what to say before generating text autoregressively with GPT-2 Large. It produces much better text (MAUVE 94.6 vs 85.2 for GPT-2 Large) but is slow because each diffusion step runs the full 770M-parameter GPT-2 backbone. My CS 5220 (parallel computing) project optimizes this inference pipeline using profiling, roofline analysis, custom Metal GPU kernels for Apple Silicon, and structural code changes.

## The story

Language models like GPT generate text one word at a time with no ability to plan ahead. STAR-LDM adds a "thinking" phase: 50 rounds of diffusion refinement in a continuous 768-dimensional sentence embedding space before committing to tokens. This produces dramatically better text but is slow.

We profiled the pipeline and found GPT-2 accounts for 77% of runtime. Digging deeper, we discovered the bottleneck wasn't GPU compute but software framework overhead: HuggingFace issues 6,185 operator dispatches per diffusion step, of which only 432 (7%) do actual math. 93% is bookkeeping: mask construction, cache management, tensor reshaping.

We wrote 6 custom Metal GPU kernels (1,700 lines of code). Only 2 of 6 were faster than PyTorch's defaults. The four that failed teach us where custom GPU kernels cannot help: when dispatch overhead exceeds compute (DDPM step on 9KB tensors), when vendor BLAS is already optimal (fused FFN), and when the crossover point is too narrow (decode-N attention). We also tried Picard iteration to parallelize diffusion steps and torch.compile on MPS, both of which made things worse.

The biggest win came from two structural changes: (1) rewriting the GPT-2 forward pass in 40 lines of plain PyTorch that strip framework abstractions (3.6x faster per call), and (2) KV-cache reuse with pre-allocated buffers that make per-step cost O(1) instead of O(prefix_length).

We evaluated on real C4 validation text across prefix lengths from 16 to 1000 tokens. At 512-token prefixes with 50 diffusion steps, the optimized pipeline achieves 4.93x end-to-end speedup. The speedup grows with prefix length because the baseline cost scales linearly while ours stays flat.

## Poster layout guidance from professor

- ~25% introduction/background on the architecture  
- ~50% what we accomplished and results
- ~25% future work (what we'll do between poster session and final report)
- Less text is preferred. Figures and tables dominate.

## Background reading

The STAR-LDM paper: https://arxiv.org/abs/2602.20528

Key related work:
- Roofline model (Williams, Waterman, Patterson, CACM 2009): the performance analysis framework we use
- FlashAttention (Dao et al., NeurIPS 2022): canonical GPU kernel fusion for attention, designed for long sequences. We target the opposite regime (8 tokens)
- DeepSpeed Inference (Aminabadi et al., SC 2022): identified kernel launch overhead as the bottleneck at small batch sizes, matching our finding
- PyTorch 2 / TorchDynamo (Ansel et al., ASPLOS 2024): torch.compile designed to address dispatch overhead via graph capture. Works on CUDA but fails on MPS
- LLM Inference Unveiled (Yuan et al., 2024): roofline analysis applied to LLM inference, showing decode is memory-bound. We extend this to show micro-workloads are dispatch-latency-bound
- PagedAttention / vLLM (Kwon et al., SOSP 2023): KV-cache memory management for serving. Our KV-cache reuse across diffusion steps is analogous

## Architecture details

STAR-LDM (956M params total) follows a Stop-Think-AutoRegress pipeline:

**Stop:** Encode the input prefix through GPT-2 Large. Cache key-value projections.

**Think:** 50-step diffusion loop. Each step:
1. Noised embedding z_t (768-dim) enters the Soft Prompt Generator (SPG)
2. SPG: Linear(768→3072) → reshape to (8, 384) → Linear(384→1024) → 6 transformer layers (dim 1024, 16 heads, non-causal, FiLM time-conditioning) → Linear(1024→1280). Outputs 8 soft prompt tokens in GPT-2's embedding space.
3. The 8 soft prompts go through GPT-2 Large (36 layers, dim 1280, 20 heads, 770M params) attending to the cached prefix KV.
4. GPT-2 hidden states are concatenated with the soft prompts (2560-dim) and fed to the Score Network Head (SNH): Linear(2560→1024) → 6 transformer layers → Linear→reshape→Linear to produce a 768-dim v-prediction.
5. The v-prediction is converted to predicted clean embedding x_0 and noise eps, then a DDPM update produces z_{t-1}.
6. Repeat 50 times.

**AutoRegress:** The final denoised soft prompts condition GPT-2 for token-by-token text generation (32-64 new tokens).

Quality results from the original paper:
| Model | Parameters | MAUVE (higher = better) |
|---|---|---|
| GPT-2 Large | 770M | 85.2 |
| GPT-2 XL | 1.5B | 86.6 |
| Pythia 1.4B | 1.4B | 84.8 |
| **STAR-LDM** | **956M** | **94.6** |

MAUVE measures distributional similarity to human text. STAR-LDM's +8 point gap over GPT-2 XL is substantial.

## All experimental data

All experiments run on Apple M4 Max (40-core GPU, 128 GB unified memory, 546 GB/s bandwidth), PyTorch 2.10, MPS backend.

### 1. Baseline profiling (50 C4 validation prompts, unoptimized, 50 steps)

| Component | Mean (ms) | % of total | Calls per generation |
|---|---|---|---|
| GPT-2 forward (diffusion loop) | 884 | 38.7% | 50 |
| GPT-2 generate (AR decode) | 879 | 38.5% | 1 |
| SoftPromptGenerator | 208 | 9.1% | 51 |
| ScoreNetHead | 211 | 9.2% | 50 |
| DDPM step + noise schedule | 31 | 1.4% | 50 |
| Other (embedding, concat, etc.) | 70 | 3.1% | — |
| **Total** | **2,282** | **100%** | |

GPT-2 backbone = 77.2% of total inference time. The micro-transformers (SPG + ScoreNet) = 18.3%. Diffusion arithmetic = 2.1%.

### 2. Operator dispatch analysis (torch.profiler, per v-prediction call)

| Metric | HuggingFace | Streamlined (ours) |
|---|---|---|
| Total ATen dispatches | 6,185 | 432 |
| Useful computation ops | 432 | 432 |
| Framework overhead ops | 5,753 | 0 |
| Average per-dispatch cost | ~3.4 μs | ~3.4 μs |
| Dispatch overhead per call | ~21 ms | ~1.5 ms |

Top framework overhead consumers in HuggingFace:
- view/empty/copy_ (tensor management): 16,090 calls
- addmm (linear projections): 1,440 calls
- where/arange/le (attention mask construction): 806 calls
- scaled_dot_product_attention (with internal overhead): 360 calls

The 432 essential ops: 36 layers x 12 ops each (2 layer norms, QKV projection, attention, output projection, residual, FFN up, GELU, FFN down, residual, plus a few extras).

### 3. Roofline analysis

Hardware: M4 Max — 14 TFLOPS FP32 peak, 546 GB/s memory bandwidth, ridge point 25.6 FLOP/byte.

| Operation | Time (μs) | Arith. Intensity (F/B) | Achieved GFLOP/s | Roofline Ceiling GFLOP/s | Efficiency |
|---|---|---|---|---|---|
| RMSNorm+FiLM (JIT) | 27 | 0.74 | 2.1 | 402 | 0.5% |
| RMSNorm+FiLM (Metal) | 13 | 0.74 | 4.5 | 402 | 1.1% |
| Tiny Attention (JIT) | 95 | 2.28 | 3.2 | 1,245 | 0.3% |
| Tiny Attention (Metal) | 40 | 2.28 | 7.4 | 1,245 | 0.6% |
| DDPM Step (JIT) | 41 | 1.25 | 0.4 | 682 | 0.1% |
| GPT-2 decode-8 (KV cached, pfx=64) | 95,560 | 4.00 | 120 | 2,184 | 5.5% |
| SoftPromptGenerator (6 layers) | 2,775 | 4.04 | 440 | 2,206 | 19.9% |

Key finding: micro-transformer operations achieve less than 1% of the roofline ceiling. The standard roofline model predicts they should sustain hundreds of GFLOP/s based on their arithmetic intensity. The gap is dispatch latency: each tensor is so small (8-128 KB) that the fixed cost of launching a GPU kernel exceeds the computation time. We term this the "dispatch-latency-bound" regime — a third regime beyond the classical compute-bound and memory-bound categories.

### 4. What we built — optimization methods

**Method A: Streamlined GPT-2 Forward Pass**
Replaced HuggingFace's GPT2LMHeadModel.forward() with 40 lines of PyTorch. Directly iterates over transformer blocks: F.layer_norm → F.linear (QKV) → F.scaled_dot_product_attention → F.linear (out proj) → F.layer_norm → F.linear (FFN up) → F.gelu → F.linear (FFN down). No mask construction, no DynamicCache, no tensor format conversions.
- Dispatch reduction: 6,185 → 432 (93%)
- Per-call speedup: 3.6x on the component that is 70% of runtime
- Correctness: max absolute difference vs HuggingFace < 1e-5

**Method B: KV-Cache Amortization with Pre-Allocated Buffers**
The unoptimized pipeline re-processes all prefix tokens at every diffusion step. We compute prefix KV projections once and reuse them. Critical implementation detail: pre-allocate contiguous KV buffers of shape (prefix_len + 8) for each of 36 layers. Write new soft-prompt KV into fixed slots each step. The naive approach using torch.cat allocates 1,800 new tensors per generation (36 layers × 50 steps), causing 10x slowdown on long prefixes due to MPS memory fragmentation.
- Per-step cost: O(1) vs O(prefix_length)
- Memory allocation in diffusion loop: 0 (vs 1,800 per generation)

**Method C: Custom Metal Compute Shaders**
1,700 lines of Metal Shading Language + Objective-C++ dispatch code. Six kernels:

| # | Kernel | Lines | What it fuses | Per-op speedup | End-to-end status |
|---|---|---|---|---|---|
| 1 | RMSNorm+FiLM | 278 | L2 norm + scale + gamma + FiLM affine in one threadgroup | 1.69x | Used in production |
| 2 | Tiny 8-token Attention | 270 | QK-norm + 8×8 attention matrix in registers + softmax, no tiling | 2.39x | Used in production |
| 3 | DDPM Denoising Step | 172 | 17 elementwise ops (variance interpolation, log/exp, noise mixing) | 0.35x (slower) | Disabled — dispatch > compute on 9KB tensor |
| 4 | Decode-N Attention | 337 | Online softmax streaming for few-query, moderate-KV pattern | 1.97x at S<40, slower above | Not integrated — crossover too narrow |
| 5 | Fused FFN | 341 | LayerNorm + GEMM + GELU + GEMM with intermediate in shared memory | 0.01x (127x slower) | Failed — can't beat Apple's BLAS |
| 6 | Speculative Verify | 302 | Softmax over 50K vocab + accept/reject for K=4 candidates | 0.13x (slower) | Not used |

All six compile correctly and produce numerically verified results. Two deliver real speedups. Four are informative negative results that demonstrate the boundaries of where GPU kernel fusion helps.

**Method D: Picard Iteration for Parallel Diffusion (failed)**
Following ParaDiGMS (Shih et al., NeurIPS 2023), we implemented fixed-point iteration to parallelize diffusion steps. Instead of sequential z_50 → z_49 → ... → z_0, guess the entire trajectory and refine in parallel. After 20 iterations, cosine similarity to sequential reference reached only 0.78 — far from convergence. The denoising function in STAR-LDM (which includes GPT-2 Large) has a much higher Lipschitz constant than the U-Net denoisers Picard iteration was designed for. Not all diffusion models are amenable to parallel sampling.

**Method E: torch.compile on MPS (failed)**
Tested both inductor and aot_eager backends on the GPT-2 forward pass. Both produced 2-3x slowdowns. The MPS backend does not benefit from the graph-level optimizations (kernel fusion, memory planning) that make torch.compile effective on CUDA. The inductor backend warned "Not enough SMs to use max_autotune_gemm mode," applying CUDA-specific heuristics that don't translate to Apple's GPU.

**Method F: Fast AR Generation**
Replaced HuggingFace's .generate() with a custom autoregressive decode loop using the same streamlined forward approach. Manages its own KV-cache incrementally. Includes top-p sampling and repetition penalty. Per-token decode: 11ms vs 22ms with HuggingFace (2x faster).

### 5. Main result: prefix length scaling on real C4 validation text

Evaluated on real documents from the C4 validation split (allenai/c4). 10 C4 documents per prefix length bucket, 3 runs each = 30 measurements per data point. All numbers are mean ± 95% CI.

**20 Diffusion Steps:**

| Prefix (tokens) | Baseline (ms) | Optimized (ms) | Speedup | n |
|---|---|---|---|---|
| 16 | 898 ± 9 | 740 ± 5 | 1.21x | 30 |
| 64 | 1,102 ± 18 | 776 ± 7 | 1.42x | 30 |
| 128 | 1,464 ± 36 | 831 ± 12 | 1.76x | 30 |
| 256 | 2,161 ± 28 | 888 ± 7 | 2.43x | 30 |
| 512 | 3,428 ± 62 | 1,013 ± 11 | 3.38x | 30 |
| 768 | 5,100 ± 37 | 1,186 ± 6 | 4.30x | 30 |
| 1000 | 6,558 ± 90 | 1,107 ± 20 | 5.92x | 30 |

**50 Diffusion Steps:**

| Prefix (tokens) | Baseline (ms) | Optimized (ms) | Speedup | n |
|---|---|---|---|---|
| 16 | 1,707 ± 27 | 1,235 ± 16 | 1.38x | 30 |
| 64 | 2,235 ± 34 | 1,227 ± 8 | 1.82x | 30 |
| 128 | 2,876 ± 23 | 1,281 ± 13 | 2.25x | 30 |
| 256 | 4,389 ± 26 | 1,347 ± 10 | 3.26x | 30 |
| 512 | 7,361 ± 27 | 1,492 ± 9 | 4.93x | 30 |
| 768 | 12,595 ± 676 | 2,469 ± 468 | 5.10x | 30 |
| 1000 | 18,355 ± 950 | 4,273 ± 840 | 4.30x | 30 |

Note: at 768 and 1000 tokens with 50 steps, both baseline and optimized show high variance (676-950ms CI) due to MPS memory pressure on some C4 documents. The clean scaling holds through 512 tokens. The headline numbers are up to **5.92x at 20 steps** and **4.93x at 50 steps** (at 512 tokens where results are clean).

The pattern: baseline cost is O(prefix_length × steps) because it re-processes the entire prefix at every step. Optimized cost is approximately O(steps) because we process only 8 tokens per step. The gap grows linearly with prefix length.

### 6. Per-step component breakdown (with all optimizations active)

Measured at prefix=64, 50 iterations each:

| Component | Per-step (ms) | % of step | x50 steps |
|---|---|---|---|
| SPG (6 layers, fused Metal) | 1.95 | 11.2% | 97 ms |
| Fast GPT-2 (36 layers) | 13.26 | 76.7% | 663 ms |
| ScoreNet (6 layers, fused Metal) | 1.73 | 10.0% | 87 ms |
| DDIM step | 0.36 | 2.1% | 18 ms |
| **Total per step** | **17.29** | | **865 ms** |

Inside one GPT-2 layer (prefix=64):
| Op | Time (ms) | % of layer |
|---|---|---|
| SDPA attention | 0.117 | 35.6% |
| FFN (up + GELU + down) | 0.126 | 38.4% |
| QKV projection | 0.039 | 12.0% |
| Output projection | 0.025 | 7.7% |
| LayerNorm x2 | 0.021 | 6.3% |
| **Per layer** | **0.328** | |
| **x36 layers** | **11.8** | |

### 7. Metal kernel microbenchmarks

Tested with correct STAR-LDM dimensions, 100 warmup + 1000 timed iterations:

| Kernel | Dimensions | JIT (ms) | Metal (ms) | Speedup | Invocations/gen |
|---|---|---|---|---|---|
| RMSNorm+FiLM | (1, 8, 1024) | 0.027 | 0.016 | 1.69x | 1,200 |
| Tiny Attention | (1, 16, 8, 64) | 0.099 | 0.042 | 2.39x | 600 |
| DDPM Step | (1, 768) | 0.040 | 0.115 | 0.35x | 49 |
| Decode-N Attn | (1, 20, 8, 64) vs KV S=13 | 0.021 | 0.011 | 1.97x | N/A |
| Fused FFN | (8, 1280→5120→1280) | 0.124 | 15.77 | 0.01x | N/A |
| Spec Verify | K=4, V=50257 | 0.939 | 7.214 | 0.13x | N/A |

Projected savings from the two successful kernels:
- RMSNorm+FiLM: 1,200 calls × (0.027 - 0.016) = 13.2 ms saved per generation
- Tiny Attention: 600 calls × (0.099 - 0.042) = 34.2 ms saved per generation
- Total kernel savings: ~47 ms per generation

### 8. Batch size scaling (50 steps, all optimizations)

| Batch Size | Total (ms) | Per-sample (ms) | Throughput (samples/s) | Scaling |
|---|---|---|---|---|
| 1 | 1,339 | 1,339 | 0.75 | 1.00x |
| 2 | 1,926 | 963 | 1.04 | 1.39x |
| 4 | 2,987 | 747 | 1.34 | 1.79x |
| 8 | 5,895 | 737 | 1.36 | 1.82x |

Throughput plateaus at batch 4-8. The GPU saturates on this workload at B=4 on M4 Max.

### 9. Decode-N attention kernel scaling across KV lengths

Custom Metal kernel for GPT-2's "few queries attending to KV cache" pattern:

| KV Length | SDPA (ms) | Metal (ms) | Speedup |
|---|---|---|---|
| 13 (prefix=5) | 0.021 | 0.011 | 1.97x |
| 40 (prefix=32) | 0.016 | 0.016 | 1.01x |
| 72 (prefix=64) | 0.011 | 0.021 | 0.54x |
| 128 | 0.013 | 0.035 | 0.36x |
| 256 | 0.016 | 0.068 | 0.24x |

The kernel wins at short KV (dispatch overhead of SDPA dominates) but loses at longer KV (Apple's tiled matmul amortizes better). Crossover at ~40 tokens. End-to-end integration showed zero net improvement because Python wrapper overhead consumed the 18ms of per-call savings.

### 10. Picard iteration convergence data

Tested at 20 diffusion steps, measuring embedding cosine similarity to the sequential DDIM reference:

| Picard Iterations | Time (ms) | Cosine Similarity to Sequential |
|---|---|---|
| 2 | 534 | 0.01 |
| 3 | 588 | 0.07 |
| 5 | 692 | 0.06 |
| 10 | 974 | 0.24 |
| 15 | 1,245 | 0.50 |
| 20 | 1,599 | 0.78 |
| Sequential (reference) | 841 | 1.00 |

By the time Picard approaches convergence (~20 iterations), it takes 2x longer than sequential. The denoising function is too nonlinear for efficient fixed-point iteration.

## Summary of all methods attempted

| Method | Type | Result | Impact |
|---|---|---|---|
| Streamlined GPT-2 forward | Software restructuring | 3.6x per-call on 70% of pipeline | Primary contributor |
| KV-cache + pre-alloc buffers | Structural optimization | O(1) vs O(n) per-step, up to 4.93x E2E | Primary contributor |
| Metal RMSNorm+FiLM kernel | GPU kernel (Metal) | 1.69x per-op | Minor E2E contribution |
| Metal tiny attention kernel | GPU kernel (Metal) | 2.39x per-op | Minor E2E contribution |
| Metal DDPM step kernel | GPU kernel (Metal) | 0.35x (slower) | Negative result |
| Metal fused FFN kernel | GPU kernel (Metal) | 0.01x (127x slower) | Negative result |
| Metal decode-N attention | GPU kernel (Metal) | 1.97x at S<40 only | Negative result |
| Metal spec verify kernel | GPU kernel (Metal) | 0.13x (slower) | Negative result |
| Fast AR generation | Software restructuring | 2x per-token decode | Moderate contributor |
| Picard parallel diffusion | Algorithmic | Did not converge | Negative result |
| torch.compile on MPS | Framework tool | 2-3x slower | Negative result |
| fp16 GPT-2 forward | Precision reduction | ~4% improvement | Minor contributor |

## Future work (for between poster session and final report)

1. **Cross-platform CUDA comparison:** Run identical benchmarks on NVIDIA A100. Does the streamlined forward help on CUDA? Does torch.compile with CUDA Graphs succeed where MPS failed? This isolates which bottlenecks are hardware-specific vs framework-general.

2. **Quality validation:** Generate 5,000 C4 continuations with both pipelines (matching the original paper's eval: 32-token prefixes, 64-token continuations). Compute MAUVE and perplexity to verify optimizations preserve output quality. Critical because our streamlined forward drops causal masking on the 8 soft prompts.

3. **Model scheduling:** Profile v-prediction error at each noise level. At early (high noise) and late (low noise) diffusion steps, GPT-2 may add minimal value. Replace with cheap SPG+ScoreNet-only path (~3ms vs ~12ms per step). Training-free, potential 30-40% further speedup.

4. **Consistency distillation (stretch goal):** Train student score network to predict clean embedding in 1-4 steps instead of 50, following Latent Consistency Models (Luo et al., ICLR 2024). STAR-LDM's 768-dim embedding space is smoother than pixel space. At 4 steps: projected total ~550ms, faster than GPT-2 Large with better quality.

## Hardware

Apple M4 Max, 40-core GPU, 128 GB unified memory, 546 GB/s memory bandwidth. PyTorch 2.10, MPS backend. Metal Shading Language for custom GPU kernels. Metal kernels compiled via torch.utils.cpp_extension with Objective-C++ dispatch through the MPS command stream API.

## Codebase

1,700 lines of Metal + Obj-C++ kernel code. ~1,000 lines of optimized PyTorch (streamlined forward, fast generate, KV-cache management). ~3,300 lines of profiling, benchmarking, and evaluation scripts. All at https://github.com/srkvatsa/STAR-LDM/tree/perf/mps-compat-and-profiling
