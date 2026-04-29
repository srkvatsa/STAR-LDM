/**
 * Fused RMSNorm + FiLM conditioning — Metal compute shader.
 *
 * Two-pass within single dispatch:
 *   Pass 1: Parallel reduction for L2 norm across D=768
 *   Pass 2: Fused normalize + scale + gamma + FiLM modulation
 *
 * Output: x_norm * gamma * (film_scale + 1) + film_shift
 *
 * Threadgroup: one per (batch, token) pair
 * Threads cooperate on the D-dimension reduction.
 *
 * Tensor layout:
 *   x:          (B, L, D) contiguous float
 *   gamma:      (D,) learned RMSNorm scale
 *   film_scale: (B, 1, D) or (B, L, D) — broadcast handled by caller
 *   film_shift: (B, 1, D) or (B, L, D)
 */

#include <metal_stdlib>
using namespace metal;

// Shared memory for parallel reduction
kernel void rmsnorm_film_kernel(
    device const float* x           [[buffer(0)]],
    device const float* gamma       [[buffer(1)]],  // (D,)
    device const float* film_scale  [[buffer(2)]],  // (N, D) where N = B*L or B*1
    device const float* film_shift  [[buffer(3)]],  // (N, D)
    device float*       out         [[buffer(4)]],
    constant uint&      D           [[buffer(5)]],
    constant float&     dim_scale   [[buffer(6)]],  // sqrt(D)
    constant uint&      film_stride [[buffer(7)]],  // L if film is (B,L,D), 1 if (B,1,D)
    uint tgid   [[threadgroup_position_in_grid]],    // (batch * L + token)
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {
    // tgid = b * L + l  (one threadgroup per (batch, token))
    uint row = tgid;  // row in the (B*L, D) view of x
    uint base = row * D;

    // --- Pass 1: compute ||x||_2 via parallel reduction ---
    float partial_sum = 0.0f;
    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        partial_sum += val * val;
    }
    shared[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm_val = max(sqrt(shared[0]), 1e-8f);

    // --- Pass 2: normalize, scale, gamma, FiLM ---
    // Determine film row: if film_stride == 1, film row = b (broadcast over L)
    // We compute: b = row / L (but we need L). We encode film addressing
    // by passing the film pointer pre-indexed or using film_stride.
    // film index: for (B,1,D) case, film_row = row / L * 1 = b
    // for (B,L,D) case, film_row = row
    // Caller sets film_stride = L for (B,1,D) broadcast, or 1 for (B,L,D)
    // Actually: film is always passed as (total_rows, D) from Python side.
    // We just index film_row = row directly (Python handles expansion).

    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        float normalized = val / norm_val * dim_scale;
        float scaled = normalized * gamma[d];
        out[base + d] = scaled * (film_scale[base + d] + 1.0f) + film_shift[base + d];
    }
}

// RMSNorm without FiLM (for layers without time conditioning)
kernel void rmsnorm_kernel(
    device const float* x           [[buffer(0)]],
    device const float* gamma       [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      D           [[buffer(3)]],
    constant float&     dim_scale   [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {
    uint base = tgid * D;

    // Pass 1: L2 norm reduction
    float partial_sum = 0.0f;
    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        partial_sum += val * val;
    }
    shared[tid] = partial_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm_val = max(sqrt(shared[0]), 1e-8f);

    // Pass 2: normalize + scale
    for (uint d = tid; d < D; d += tg_sz) {
        float val = x[base + d];
        out[base + d] = val / norm_val * dim_scale * gamma[d];
    }
}
