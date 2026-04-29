# Poster Content for Claude Web

## Instructions for Claude Web

Create a 40 inches wide x 30 inches tall academic research poster (landscape orientation) for a CS class project. The style should be clean and modern, similar to a top ML conference poster. Use Cornell red (#B31B1B) as the primary accent color. The poster will be printed on glossy paper at 30x40 inches.

Layout: 4 columns. Approximately 25% introduction/background, 50% results, 25% future work. Prefer figures and tables over dense text. Every section should be readable from 4-6 feet away.

---

## HEADER

**Title:** Inference Optimization for Hybrid Diffusion-Autoregressive Language Models: Profiling, Fusion, and Framework Overhead

**Author:** Srivatsa Kundurthy, Cornell University

**Course:** CS 5220: Applied High-Performance and Parallel Computing, Spring 2026

**Hardware:** Apple M4 Max, 40-core GPU, 128 GB unified memory

---

## SECTION 1: BACKGROUND (Column 1)

### What is STAR-LDM?

STAR-LDM (COLM 2025) is a language model that plans before it writes. It uses a 50-step diffusion process to refine a "mental sketch" (a 768-dimensional sentence embedding), then converts that plan into actual text using GPT-2 Large.

This planning mechanism produces substantially better text than standard models:

| Model | Parameters | MAUVE Score |
|---|---|---|
| GPT-2 Large | 770M | 85.2 |
| GPT-2 XL | 1.5B | 86.6 |
| **STAR-LDM** | **956M** | **94.6** |

MAUVE measures how similar generated text is to human writing (higher = better). STAR-LDM achieves an 8-point improvement over GPT-2 XL despite being a smaller model.

### The Inference Pipeline

STAR-LDM follows a "Stop-Think-AutoRegress" pipeline:

1. **Stop:** Encode the input prefix through GPT-2 Large (36 layers, 770M parameters).
2. **Think:** Run 50 diffusion steps. At each step:
   - Noised embedding z_t (768-dim) → Soft Prompt Generator (6 transformer layers) → 8 soft prompt tokens (1280-dim)
   - Feed 8 soft prompts through GPT-2 Large (36 layers)
   - GPT-2 output → Score Network Head (6 transformer layers) → v-prediction (768-dim)
   - Apply DDPM denoising update → z_{t-1}
3. **AutoRegress:** Use final soft prompts to condition GPT-2 for token-by-token text generation.

### The Problem

Each of the 50 diffusion steps invokes the full 770M-parameter GPT-2 backbone. This makes STAR-LDM approximately 2.3x slower than GPT-2 Large alone. The original authors acknowledge their implementation is "not optimized for inference speed."

---

## SECTION 2: PROFILING & ANALYSIS (Column 2, top half)

### Where Does the Time Go?

Profiled on 50 C4 validation prompts, M4 Max, 50 diffusion steps:

| Component | % of Time |
|---|---|
| GPT-2 forward (diffusion loop) | 38.7% |
| GPT-2 generate (AR decode) | 38.5% |
| SoftPromptGenerator | 9.1% |
| ScoreNetHead | 9.2% |
| Other (DDPM step, noise schedule) | 4.5% |

**GPT-2 accounts for 77.2% of total inference time.**

### The Real Bottleneck: Framework Overhead

Using torch.profiler, we counted the operator dispatches inside a single diffusion step:

- **HuggingFace GPT-2 forward: 6,185 ATen operator dispatches per call**
- **Only 432 (7%) perform actual computation**
- The other 93% are framework overhead: attention mask construction, cache management, tensor reshaping, type checking
- At ~3.4 microseconds per dispatch, this overhead alone accounts for ~21ms per call

**INCLUDE A BAR CHART: Two bars. Left bar "HuggingFace" = 6,185 dispatches (red). Right bar "Streamlined (ours)" = 432 dispatches (green). Label: "93% reduction." Y-axis: "ATen Operator Dispatches per Call"**

### Roofline Analysis

We computed arithmetic intensity and achieved throughput for each operation against the M4 Max hardware ceiling (14 TFLOPS compute, 546 GB/s bandwidth):

| Operation | Achieved GFLOP/s | Roofline Ceiling | Efficiency |
|---|---|---|---|
| RMSNorm+FiLM (Metal) | 4.5 | 402 | 1.1% |
| Tiny Attention (Metal) | 7.4 | 1,245 | 0.6% |
| DDPM Step (JIT) | 0.4 | 682 | 0.1% |
| GPT-2 decode-8 | 120 | 2,184 | 5.5% |
| SoftPromptGenerator | 440 | 2,206 | 19.9% |

Micro-transformer operations achieve less than 1% of the roofline ceiling. They are not compute-bound or memory-bound. They are **dispatch-latency-bound**: the tensors are so small (8-128 KB) that the fixed cost of launching a GPU kernel exceeds the actual computation time.

**INCLUDE A LOG-LOG ROOFLINE PLOT: X-axis "Arithmetic Intensity (FLOP/byte)", Y-axis "Throughput (GFLOP/s)". Black line showing roofline (slope = 546 GB/s, plateau = 14,000 GFLOP/s). Dots for each operation labeled. Shade the region below 10 GFLOP/s in light red and label it "Dispatch-latency-bound regime (<1% ceiling)".**

---

## SECTION 3: OPTIMIZATIONS & RESULTS (Column 2 bottom + Column 3)

### Optimization A: Streamlined GPT-2 Forward

Replaced HuggingFace's GPT2LMHeadModel.forward() with 40 lines of PyTorch that perform only the essential computation. Eliminates mask construction, cache management, tensor format conversions.

**Result: 6,185 → 432 dispatches. 3.6x faster per-call on the component that is 70% of total runtime.**

### Optimization B: KV-Cache with Pre-Allocated Buffers

Compute prefix key-value projections once, reuse across all 50 diffusion steps. Pre-allocate contiguous KV buffers to eliminate torch.cat allocation (1,800 allocations per generation → 0).

**Result: Per-step cost changes from O(prefix_length) to O(1).**

### Optimization C: Metal Compute Shaders (1,700 lines of code)

Six custom Apple Metal GPU kernels. Two deliver speedups, four are informative negative results:

| Kernel | What It Does | Speedup |
|---|---|---|
| ✓ RMSNorm+FiLM | Fuse 3 dispatches → 1 | 1.69x |
| ✓ Tiny Attention | Register-resident 8×8, no tiling | 2.39x |
| ✗ DDPM Step | Fuse 17 dispatches → 1 | 0.35x (slower) |
| ✗ Fused FFN | LN+GEMM+GELU+GEMM | 0.01x (slower) |
| ✗ Decode-N Attn | Online softmax streaming | wins only at KV<40 |
| ✗ Spec Verify | Softmax + accept/reject | 0.13x (slower) |

DDPM step loses because dispatch overhead exceeds compute on 9KB tensors. Fused FFN loses because Apple's BLAS is already optimal. 

### Main Result: Prefix Length Scaling

**This is the central result. It should be the largest and most prominent figure on the poster.**

**INCLUDE A LINE CHART: X-axis "Prefix Length (tokens)" with values 16, 64, 128, 256, 512, 768, 900, 1000. Y-axis "End-to-End Latency (ms)". Two lines: Red line "Baseline (HuggingFace)" rising steeply. Green line "Optimized (this work)" staying nearly flat. Shade the area between the lines. Label speedup values at each point. Include both 20-step and 50-step data (either as two separate panels or as solid/dashed lines).**

Data for the chart:

**50 Diffusion Steps (5 runs, 95% CI):**

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

**20 Diffusion Steps (5 runs, 95% CI):**

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

**Also include a speedup bar chart next to the main plot: bars showing speedup at each prefix length, showing the increasing trend from 1.33x to 9.46x.**

### Why the Speedup Scales

The baseline re-processes the entire prefix at every diffusion step. Cost is proportional to prefix_length × steps. Our optimized pipeline processes only 8 tokens per step regardless of prefix length. Cost is approximately constant. The gap grows linearly with both prefix length and step count.

At 1000 tokens, 50 steps: baseline takes 14.4 seconds, optimized takes 1.5 seconds.

### Negative Results

These are important and should be briefly mentioned:

- **Picard parallel diffusion:** Attempted to run diffusion steps in parallel via fixed-point iteration. Convergence too slow because GPT-2's denoising function is too nonlinear. After 20 iterations, only 0.78 cosine similarity to sequential reference.
- **torch.compile on MPS:** Both inductor and aot_eager backends made GPT-2 2-3x slower on Apple Silicon.
- **Fused FFN Metal kernel:** 127x slower than Apple's optimized BLAS. Cannot out-kernel the vendor's matrix multiply.

---

## SECTION 4: FUTURE WORK (Column 4)

### A. Cross-Platform CUDA Comparison
Run identical benchmarks on NVIDIA A100. Test whether framework overhead is MPS-specific or general. Does torch.compile with CUDA Graphs succeed where MPS failed?

### B. Quality Validation
Generate 5,000 C4 validation continuations with both pipelines (matching the original paper's evaluation setup: 32-token prefixes, 64-token continuations). Compute MAUVE and perplexity to verify optimizations preserve output quality. Critical because our streamlined forward drops causal masking on the 8 soft prompts.

### C. Model Scheduling
Profile v-prediction error at each noise level. At early and late diffusion steps, GPT-2 may add minimal value. Replace with cheap SPG+ScoreNet-only path (~3ms vs ~12ms per step). Training-free, potential 30-40% further speedup.

### D. Consistency Distillation (Stretch Goal)
Train a student score network to predict the clean embedding in 1-4 steps instead of 50, following Latent Consistency Models (Luo et al., ICLR 2024). STAR-LDM's 768-dim embedding space is smoother than pixel space, suggesting few-step distillation should be effective. At 4 steps, projected total latency is ~550ms — faster than GPT-2 Large with better quality.

---

## KEY TAKEAWAY BOX

**The primary bottleneck in hybrid diffusion-AR inference is framework dispatch overhead, not GPU kernel performance.**

HuggingFace's GPT-2 issues 14x more operator dispatches than the core computation requires. A 40-line PyTorch rewrite that strips framework abstractions gives a larger speedup (3.6x on 70% of the pipeline) than six custom Metal GPU kernels combined.

Roofline analysis reveals a dispatch-latency-bound regime where micro-transformer operations achieve <1% of hardware ceiling — a third regime beyond compute-bound and memory-bound that standard performance modeling does not capture.

---

## REFERENCES (small text at bottom)

- Lovelace et al., "STAR-LDM," COLM 2025
- Dao et al., "FlashAttention," NeurIPS 2022
- Aminabadi et al., "DeepSpeed Inference," SC 2022
- Williams et al., "Roofline Model," CACM 2009
- Yuan et al., "LLM Inference Unveiled," 2024
- Ansel et al., "PyTorch 2," ASPLOS 2024
- Kwon et al., "PagedAttention / vLLM," SOSP 2023

---

## FIGURE FILES AVAILABLE

All figures are pre-generated as PNG and PDF in the figures/ directory:
- figures/fig1_profiling_pie.png — runtime breakdown pie chart
- figures/fig2_prefix_scaling.png — latency vs prefix length (needs regeneration with new data)
- figures/fig3_roofline.png — roofline plot with dispatch-latency-bound region
- figures/fig4_dispatch_count.png — 6,185 vs 432 bar chart
- figures/fig5_kernel_microbench.png — Metal kernel speedups
- figures/fig0_architecture.png — STAR-LDM architecture diagram
