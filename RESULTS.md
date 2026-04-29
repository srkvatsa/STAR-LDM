# STAR-LDM Inference Optimization: Complete Results

**Hardware**: Apple M4 Max, 40-core GPU, 128 GB unified memory, 546 GB/s bandwidth  
**Model**: STAR-LDM (956M params): GPT-2 Large (770M) + SoftPromptGenerator (~80M) + ScoreNetHead (~80M)  
**Checkpoint**: Real trained weights (fineweb sample-10BT, 250K steps)  
**Date**: April 10, 2026  

---

## 1. Baseline Profiling

Profiled the unoptimized STAR-LDM pipeline on 50 C4 validation prompts (varying prefix lengths, 50 diffusion steps, DDPM sampler). Per-generation breakdown:

| Component | Mean (ms) | % of Total | Calls/gen | Per-call (ms) |
|---|---|---|---|---|
| GPT-2 forward (diffusion) | 884.1 | 38.7% | 50 | 17.7 |
| GPT-2 generate (AR) | 878.8 | 38.5% | 1 | 878.8 |
| SoftPromptGenerator | 207.8 | 9.1% | 51 | 4.1 |
| ScoreNetHead | 210.6 | 9.2% | 50 | 4.2 |
| Noise schedule | 15.9 | 0.7% | 50 | 0.3 |
| DDPM step | 15.4 | 0.7% | 49 | 0.3 |
| v-to-x0-eps conversion | 9.8 | 0.4% | 50 | 0.2 |
| Other (embed, concat, etc.) | 59.6 | 2.6% | — | — |
| **Total** | **2281.9** | **100%** | — | — |

**Key observation**: GPT-2 accounts for 77.2% of total time (38.7% diffusion loop + 38.5% AR generation). The micro-transformers (SPG + ScoreNet) account for 18.3%. The DDPM step arithmetic is only 0.7%.

Source: `results_c4_50.json` (50 prompts, varying prefix lengths 5-50 tokens)

---

## 2. End-to-End Optimization Benchmark

Five configurations tested incrementally on a single 5-token prompt ("The meaning of life is"), 50 diffusion steps, 5 runs each after 2 warmup:

| # | Configuration | Mean (ms) | Std | Speedup |
|---|---|---|---|---|
| 1 | Baseline (unfused, no KV, fp32) | 1581.5 | 37.1 | 1.00x |
| 2 | + KV-cache only | 1389.6 | 23.3 | **1.14x** |
| 3 | + Fused Metal micro-transformer ops | 1469.5 | 7.5 | **1.08x** |
| 4 | + Fused + KV-cache | 1394.3 | 20.8 | 1.13x |
| 5 | + Fused + KV-cache + Async CPU/GPU | 1426.2 | 22.9 | 1.11x |

**Additional configs** (measured separately):

| Configuration | Mean (ms) | Speedup vs baseline |
|---|---|---|
| Fused + KV-cache + fp16 | 1336 | **1.18x** |
| Fused + KV-cache + fp16 + 20 steps | 783 | **2.02x** |
| Fused + KV-cache + fp16 + 10 steps | 608 | **2.60x** |

Source: `results_e2e_50steps.json`

### Interpretation

- **KV-cache** is the largest systems-level win (1.14x). It eliminates redundant prefix recomputation: GPT-2 no longer re-processes the static prefix tokens at each of the 50 diffusion steps. Instead, key-value projections are computed once and reused.
- **Fused Metal kernels** provide 1.08x on their own (micro-transformers only, no KV-cache). They reduce kernel dispatch count from ~30 to ~10 per micro-transformer forward pass by fusing RMSNorm+FiLM and tiny attention.
- **Fused + KV-cache don't stack**: 1.13x combined vs 1.14x for KV-cache alone. This is because they target different pipeline components (micro-transformers vs GPT-2 backbone), and the GPT-2 backbone dominates.
- **Async CPU/GPU** slightly hurts performance (1.11x vs 1.13x). The overhead of the async scheduler's thread management exceeds the benefit of CPU-side noise precomputation for 768-dim tensors that are already very fast to generate.
- **fp16** gives an additional ~4% by halving memory traffic for the memory-bound GPT-2 forward.
- **Step reduction** is the largest lever: 2.02x at 20 steps, 2.60x at 10 steps. This is an algorithmic optimization (fewer diffusion iterations), not a systems one.

---

## 3. Metal Kernel Microbenchmarks

Tested with correct model dimensions: micro-transformer dim=1024, dim_head=64, 16 heads, seq_len=8, sentence_emb=768. 100 warmup + 1000 timed iterations.

| Kernel | JIT (ms) | Metal (ms) | Speedup | Invocations/gen |
|---|---|---|---|---|
| RMSNorm+FiLM (1x8x1024) | 0.027 | 0.016 | **1.69x** | 1200 |
| Tiny Attention (1x16x8x64) | 0.099 | 0.042 | **2.39x** | 600 |
| DDPM Step (1x768) | 0.040 | 0.115 | 0.35x | 49 |
| Spec Verify (K=4, V=50257) | 0.939 | 7.214 | 0.13x | — |

### Projected savings from Metal kernels

- RMSNorm+FiLM: 1200 calls x (0.027 - 0.016) = **13.2 ms saved**
- Tiny Attention: 600 calls x (0.099 - 0.042) = **34.2 ms saved**
- **Total kernel savings: ~47 ms** out of ~1580 ms baseline = **3.0%**

The observed 1.08x (7.5%) end-to-end speedup exceeds the projected 3% because the fused blocks also reduce Python-level overhead (fewer module calls, less tensor allocation).

### Why DDPM Step Metal kernel is slower

The DDPM step operates on 768-dim vectors (9 KB working set) — small enough to fit in L1 cache. The JIT-scripted version runs these elementwise ops efficiently within the MPS runtime. The custom Metal kernel adds ~0.075 ms of dispatch overhead (kernel compilation, command encoding, buffer binding) that exceeds the compute savings. **The DDPM Metal kernel is disabled in production; the JIT fallback is used.**

---

## 4. Roofline Analysis

All operations measured on M4 Max (peak 14 TFLOPS FP32, 546 GB/s memory BW, ridge point = 25.6 FLOP/byte):

| Operation | Time (us) | Arith. Intensity | Achieved GFLOPS | Achieved BW | Ceiling GFLOPS | Efficiency | Regime |
|---|---|---|---|---|---|---|---|
| RMSNorm+FiLM (JIT) | 26.8 | 0.74 F/B | 2.14 | 2.91 GB/s | 402 | 0.5% | memory |
| RMSNorm+FiLM (Metal) | 12.7 | 0.74 F/B | 4.53 | 6.15 GB/s | 402 | 1.1% | memory |
| Tiny Attention (JIT) | 94.9 | 2.28 F/B | 3.16 | 1.39 GB/s | 1245 | 0.3% | memory |
| Tiny Attention (Metal) | 40.4 | 2.28 F/B | 7.43 | 3.26 GB/s | 1245 | 0.6% | memory |
| DDPM Step (JIT) | 41.1 | 1.25 F/B | 0.37 | 0.30 GB/s | 682 | 0.1% | memory |
| DDPM Step (Metal) | 107.3 | 1.25 F/B | 0.14 | 0.11 GB/s | 682 | 0.0% | memory |
| GPT-2 decode-8 (KV, pfx=64) | 95,560 | 4.00 F/B | 119.6 | 29.9 GB/s | 2184 | 5.5% | memory |
| SoftPromptGenerator (6-layer) | 2,775 | 4.04 F/B | 439.9 | 108.9 GB/s | 2206 | 19.9% | memory |

Source: `results_roofline.json`

### Key finding: dispatch-latency-bound regime

All operations are memory-bound (below the ridge point), but the micro-transformer kernels achieve **<1% of even the memory-bound ceiling**. The standard roofline model predicts they should run at hundreds of GFLOPS (the memory-bound ceiling for their arithmetic intensity), but they achieve single-digit GFLOPS.

The gap is explained by **dispatch latency**: each kernel operates on tensors so small (8-128 KB) that the fixed cost of launching a GPU kernel (~10-100 us) dominates the actual compute time. This is a "third regime" beyond compute-bound and memory-bound that the standard roofline model does not capture.

- Micro-transformer ops: **dispatch-latency-bound** (<1% of roofline ceiling)
- GPT-2 decode-8: **moderately efficient** (5.5%, limited by sequential 36-layer execution)
- SoftPromptGenerator as a whole: **reasonably efficient** (19.9%, enough work per dispatch to amortize overhead)

---

## 5. Batch Size Scaling

Measured with fused blocks + KV-cache enabled, 50 diffusion steps:

| Batch Size | Total (ms) | Per-sample (ms) | Throughput (samples/s) | Scaling |
|---|---|---|---|---|
| 1 | 1,339 | 1,339 | 0.75 | 1.00x |
| 2 | 1,926 | 963 | 1.04 | 1.39x |
| 4 | 2,987 | 747 | 1.34 | 1.79x |
| 8 | 5,895 | 737 | 1.36 | 1.82x |

Source: `results_batch_sweep.json`

### Interpretation

Throughput scaling is sublinear and plateaus at B=4-8. Per-sample latency improves from 1339ms (B=1) to 737ms (B=8) — a 1.82x improvement — but B=4 to B=8 shows almost no per-sample gain (747 vs 737 ms). The GPU is saturated at B=4 on this workload.

The sublinearity comes from:
1. The AR generation step is sequential per sample (generates tokens one at a time)
2. At B=8, the KV cache (36 layers x 2 x 13 x 1280 x 2 bytes x 8 = ~115 MB) plus model weights (~1.5 GB at fp16) stress the memory system
3. The 50-step diffusion loop cannot be parallelized across batch elements (they share model weights)

---

## 6. Step Count vs Latency Tradeoff

With the best systems config (fused + KV-cache + fp16), single prompt:

| Diffusion Steps | Latency (ms) | Speedup vs 50 | Sample text snippet |
|---|---|---|---|
| 10 | 608 | 2.21x | "...duration to life itself. It is the way you feel..." |
| 20 | 783 | 1.72x | "...hard to discuss, and I'm sure most of us will agree..." |
| 30 | 950 | 1.42x | "...a pure light. To be Pure is to have divine power..." |
| 50 | 1,344 | 1.00x | (reference) |

The STAR-LDM paper reports that quality (MAUVE, perplexity) plateaus at ~15-20 steps with DPM-Solver. At 20 steps, the combined optimization stack delivers **2.02x total speedup** over the unoptimized 50-step baseline.

---

## 7. Decode-N Attention Kernel Analysis

Custom Metal kernel for the "few queries attending to moderate KV cache" pattern:

**Microbenchmark** (1x20 heads, N_Q=8, D=64, fp16, 1000 iterations):

| S_KV (KV length) | SDPA (ms) | Metal (ms) | Speedup | Correct |
|---|---|---|---|---|
| 13 (prefix=5) | 0.021 | 0.011 | **1.97x** | PASS |
| 40 (prefix=32) | 0.016 | 0.016 | 1.01x | PASS |
| 72 (prefix=64) | 0.011 | 0.021 | 0.54x | PASS |
| 128 (prefix=120) | 0.013 | 0.035 | 0.36x | PASS |
| 256 (prefix=248) | 0.016 | 0.068 | 0.24x | PASS |

The kernel wins for short KV (S < ~40) where dispatch overhead of SDPA dominates. At longer KV lengths, Apple's optimized SDPA with tiled matmul is superior.

**End-to-end integration**: When patched into GPT-2 via monkey-patching, the kernel shows **0 net improvement** (1333 ms vs 1358 ms). The projected savings of 18ms (0.010 ms/call x 36 layers x 50 steps) are consumed by the Python wrapper overhead.

**Implication**: Even a 2x speedup on the attention kernel doesn't move the end-to-end needle because SDPA accounts for only ~1.4% of total latency at S=13. The bottleneck is the aggregate of 36 sequential Python-dispatched layers, not any single kernel.

---

## 8. Summary of Contributions

| Contribution | Per-op Impact | E2E Impact | Status |
|---|---|---|---|
| MPS port + profiling | — | — | Complete |
| KV-cache reuse | eliminates 50x prefix recomputation | **1.14x** | Complete |
| Fused RMSNorm+FiLM (Metal) | 1.69x per-kernel | part of 1.08x | Complete |
| Fused tiny attention (Metal) | 2.39x per-kernel | part of 1.08x | Complete |
| Fused DDPM step (JIT only) | — (Metal slower) | — | Complete (Metal disabled) |
| Decode-N attention (Metal) | 1.97x at S=13 | 0x (overhead) | Complete (not integrated) |
| fp16 GPT-2 forward | ~2x memory traffic reduction | ~4% | Complete |
| Async CPU/GPU overlap | — | slightly negative | Complete (not recommended) |
| Roofline analysis | — | — | Complete |
| Batch scaling study | 1.82x throughput at B=8 | — | Complete |
| Step count sweep | — | **2.02x at 20 steps** | Complete |

**Best combined systems config**: Fused blocks + KV-cache + fp16 = **1.18x speedup**  
**Best total config**: Above + 20 steps = **2.02x speedup**
