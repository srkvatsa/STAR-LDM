# GPU Kernel Optimization for Micro-Workloads in Hybrid Diffusion-Autoregressive Language Models

## 1. Hypothesis

Hybrid diffusion-autoregressive language models like STAR-LDM [1] exhibit an unusual computational pattern during inference: a 50-iteration diffusion loop where each iteration issues dozens of GPU kernel launches on *tiny* tensors (8-token sequences, 768-dimensional embeddings). Standard deep learning primitives — FlashAttention [2], cuBLAS GEMMs, fused optimizer steps — are engineered for large batch, long-sequence workloads and carry overhead (tiling, online softmax, scheduling) that dominates compute time at this micro-scale. We hypothesize that:

1. **Kernel launch overhead is the primary bottleneck** for the micro-transformer components of STAR-LDM inference, not arithmetic throughput or memory bandwidth.
2. **Custom fused GPU kernels** targeting these fixed-size, micro-workloads (fused RMSNorm+FiLM conditioning, register-resident 8-token attention, fused DDPM denoising arithmetic) can deliver 2--5x speedups over PyTorch's default operator decomposition on individual operations, and 1.3--1.5x end-to-end.
3. **The optimal parallelization strategy differs between unified and discrete memory architectures**: on Apple Silicon (unified memory), certain micro-operations are better executed on the CPU to avoid GPU dispatch overhead entirely — a heterogeneous scheduling strategy that is impossible on discrete GPU systems.

We expect to confirm hypotheses (1) and (2) through roofline analysis and kernel microbenchmarks, and hypothesis (3) through a cross-platform comparison between Apple Metal (M4 Max) and NVIDIA CUDA.

## 2. Context

This project builds on STAR-LDM [1], a hybrid diffusion-autoregressive language model published at COLM 2025, on which one of the project members (Kundurthy) is a co-author. STAR-LDM uses a 50-step latent diffusion process to plan a sentence embedding, converts it to 8 soft prompt tokens via a micro-transformer, and feeds those into GPT-2 Large for autoregressive generation. The model is fully trained and a checkpoint is available; **this project is purely about inference-time systems optimization, not model training or architecture design.**

The optimization work may inform a future systems-focused publication (targeting NeurIPS 2026 or an EMNLP workshop), but the scope proposed here — kernel implementation, profiling, and cross-platform performance analysis — is original to this course and has not been submitted or used elsewhere.

## 3. Key Prior Work

**FlashAttention (Dao et al., NeurIPS 2022) [2].** The canonical example of hardware-aware kernel optimization for transformers. FlashAttention achieves 2--4x wall-clock speedups by fusing attention computation into a single GPU kernel with IO-aware tiling and online softmax. Critically, FlashAttention is designed for *long* sequences (thousands of tokens) — its tiling machinery becomes pure overhead when the attention matrix is 8x8 and fits entirely in registers. Our work targets this complementary regime.

**Liger-Kernel (Hsu et al., 2024) [3].** A collection of Triton-based fused kernels for transformer training and inference (fused RMSNorm, cross-entropy, SwiGLU). Liger demonstrates 20--50% memory savings and significant throughput gains from kernel fusion in standard LLM workloads. Our work extends this fusion philosophy to the novel operator patterns in diffusion-AR hybrids — specifically, the RMSNorm+FiLM conditioning pattern and coupled transcendental arithmetic in DDPM updates — which have no coverage in existing kernel libraries.

**STAR-LDM (Lovelace et al., COLM 2025) [1].** The model architecture itself. STAR-LDM's inference pipeline issues ~30 GPU kernel launches per micro-transformer forward pass (6 layers x ~5 ops), repeated 50 times in the diffusion loop. At 8 tokens and 1024 hidden dimensions, each kernel performs on the order of 10^4 FLOPs — well below the threshold where compute-bound analysis applies. This makes STAR-LDM an ideal case study for kernel launch overhead and micro-workload optimization on modern GPU architectures.

## 4. Empirical Methodology

### Phase 1: Profiling and Roofline Analysis (Week 1)

We will instrument the full STAR-LDM inference pipeline with synchronous timing barriers (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) to produce a per-component runtime breakdown (GPT-2 backbone, soft prompt generator, score network, DDPM step, noise schedule). We will construct a **roofline model** for each component, plotting achieved arithmetic intensity (FLOPs/byte) against the hardware's compute and memory bandwidth ceilings. For Apple Silicon, we will supplement with Xcode Metal System Trace captures to measure GPU utilization, command buffer gaps, and dispatch queue latency. For NVIDIA, we will use `nsys` / Nsight Compute.

### Phase 2: Custom Kernel Implementation (Weeks 2--4)

We will implement three fused GPU kernels, each in both Metal Shading Language (Apple) and Triton (NVIDIA CUDA):

- **Fused RMSNorm + FiLM conditioning.** Merges three kernel launches (normalize, project time embedding, apply affine modulation) into one. Each thread-row processes one token's 1024-dim vector. Called 1200 times per generation (12 layers x 2 norms x 50 steps).
- **Register-resident 8-token attention.** Replaces PyTorch's `scaled_dot_product_attention` with a custom kernel where Q (8x64), K (8x64), V (8x64) live entirely in registers. Full 8x8 softmax with no tiling, no shared memory. One threadgroup per (batch, head). Called 600 times per generation.
- **Fused DDPM denoising step.** Consolidates ~17 elementwise kernel launches (variance interpolation, log/exp, noise mixing) into a single kernel operating on 768-dim vectors. Called 49 times per generation. On Apple Silicon, we will also implement a CPU path using POSIX threads to exploit zero-copy unified memory access.

Additionally, we will implement **KV-cache reuse** (pure PyTorch), which restructures the GPT-2 forward pass to compute key-value projections for the static prefix tokens once and reuse them across all 50 diffusion steps — reducing GPT-2's per-step input from `prefix_len + 8` to just 8 tokens.

### Phase 3: Performance Analysis (Weeks 4--5)

We will conduct three systematic experiments:

1. **Kernel microbenchmarks.** Each fused kernel vs. its PyTorch-decomposed baseline, measured in microseconds with 1000 warm-up + 1000 timed iterations. Report achieved FLOPs, memory bandwidth, and position on the roofline.
2. **Strong scaling study.** Fix the input (single prompt, 64-token prefix, 50 diffusion steps) and measure end-to-end latency as we enable optimizations incrementally: baseline → +KV-cache → +fused RMSNorm-FiLM → +fused attention → +fused DDPM step → all combined. Also sweep batch size (1, 2, 4, 8, 16) to measure throughput scaling.
3. **Cross-platform comparison.** Run the identical benchmark suite on Apple M4 Max (Metal/MPS) and NVIDIA GPU (CUDA/Triton). Compare kernel-level timings, dispatch overhead, and memory bandwidth utilization. Analyze where unified memory enables CPU offloading strategies that are infeasible on discrete GPUs.

**Computational resources:** Apple M4 Max (128 GB unified memory, 16-core GPU, 546 GB/s bandwidth) for Metal development. For CUDA, we will use Cornell-provided GPU resources or cloud instances (A100/H100). The pretrained STAR-LDM checkpoint is available from the original authors.

## 5. Challenges and Obstacles

**Metal kernel development tooling.** Unlike CUDA (which has Triton, CUTLASS, and mature profiling with Nsight), Apple Metal for PyTorch is poorly documented and has a thin ecosystem. Custom Metal kernels require Objective-C++ dispatch code and manual buffer management through undocumented PyTorch MPS internals (`at::mps::getCurrentMPSStream`, `at::native::mps::getMTLBufferStorage`). Debugging is limited to Xcode's Metal debugger, which does not integrate with Python.

**Dispatch overhead floor.** Our hypothesis is that kernel launch overhead dominates — but if the MPS/CUDA runtime has a hard floor on dispatch latency (e.g., ~10 us per kernel), fusing 3 launches into 1 saves at most ~20 us per call. At 1200 calls, that is ~24 ms — measurable but modest relative to 1.5 s total. We may find that the real bottleneck is the GPT-2 backbone (36-layer decode attention on 8 query tokens against a long KV cache), which is a much harder optimization target requiring hooks into HuggingFace internals.

**Cross-platform parity.** Ensuring that Metal and Triton kernels produce numerically identical results (within floating-point tolerance) requires careful attention to reduction order, transcendental function implementations, and mixed-precision behavior. Metal Shading Language lacks some standard math functions (e.g., `log1p`), requiring manual workarounds.

**Roofline accuracy on Apple Silicon.** Apple does not publicly document all memory hierarchy parameters for the M4 Max GPU. We will need to rely on microbenchmarks (stream bandwidth tests, latency probes) to construct an empirical roofline rather than using manufacturer specs.

## References

[1] Lovelace, Belardi, Zalouk, Polavaram, Kundurthy, Weinberger. "STAR-LDM: Continuous Sentence Generation via Latent Diffusion with Stop, Think, and Auto-Regress." COLM 2025. arXiv:2602.20528.

[2] Dao, Fu, Ermon, Rudra, Re. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

[3] Hsu et al. "Liger-Kernel: Efficient Triton Kernels for LLM Training." 2024. github.com/linkedin/Liger-Kernel.
