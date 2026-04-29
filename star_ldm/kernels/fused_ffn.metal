/**
 * Fused FFN Kernel: Linear + GELU + Linear in one dispatch
 *
 * For GPT-2 Large's MLP: x(B,8,1280) → W_up(1280,5120) + bias → GELU → W_down(5120,1280) + bias
 *
 * Fusion benefit: the intermediate tensor (B,8,5120) = 160KB stays in threadgroup
 * shared memory instead of round-tripping through global memory. Eliminates
 * 320KB of memory traffic per layer × 36 layers = 11.5MB per v_pred call.
 *
 * Thread organization:
 *   Grid: (B * 8) threadgroups — one per output row
 *   Threadgroup: 256 threads
 *   Each threadgroup computes one row of the final output (1280 values)
 *   Phase 1: Compute x @ W_up + b_up → GELU → intermediate (5120 values in shared mem)
 *   Phase 2: Compute intermediate @ W_down + b_down → output (1280 values)
 */

#include <metal_stdlib>
using namespace metal;

// GELU approximation (tanh version, matches PyTorch's approximate='tanh')
inline float gelu_tanh(float x) {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + precise::tanh(inner));
}

/**
 * fused_ffn: x @ W_up + b_up → GELU → @ W_down + b_down
 *
 * x:      (rows, D_in)     — input, D_in = 1280
 * W_up:   (D_in, D_mid)    — up-projection weights, D_mid = 5120 (Conv1D format: in×out)
 * b_up:   (D_mid,)
 * W_down: (D_mid, D_out)   — down-projection weights, D_out = 1280
 * b_down: (D_out,)
 * out:    (rows, D_out)    — output
 */
kernel void fused_ffn(
    device const float* x       [[buffer(0)]],   // (rows, 1280)
    device const float* W_up    [[buffer(1)]],   // (1280, 5120)
    device const float* b_up    [[buffer(2)]],   // (5120,)
    device const float* W_down  [[buffer(3)]],   // (5120, 1280)
    device const float* b_down  [[buffer(4)]],   // (1280,)
    device float*       out     [[buffer(5)]],   // (rows, 1280)
    constant uint&      D_in    [[buffer(6)]],   // 1280
    constant uint&      D_mid   [[buffer(7)]],   // 5120
    constant uint&      D_out   [[buffer(8)]],   // 1280
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  tg_size   [[threads_per_threadgroup]]
)
{
    // This threadgroup handles one row (one token position)
    const uint row = tg_id;

    // Pointer to this row's input
    device const float* x_row = x + row * D_in;

    // Phase 1: Compute intermediate = GELU(x @ W_up + b_up)
    // Each thread computes D_mid/tg_size elements of the intermediate
    // Store in threadgroup shared memory
    threadgroup float intermediate[5120];  // D_mid max

    for (uint j = tid; j < D_mid; j += tg_size) {
        // Dot product: x_row[0..D_in-1] · W_up[0..D_in-1, j]
        float acc = b_up[j];
        for (uint k = 0; k < D_in; k++) {
            acc += x_row[k] * W_up[k * D_mid + j];
        }
        intermediate[j] = gelu_tanh(acc);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute output = intermediate @ W_down + b_down
    device float* out_row = out + row * D_out;

    for (uint j = tid; j < D_out; j += tg_size) {
        float acc = b_down[j];
        for (uint k = 0; k < D_mid; k++) {
            acc += intermediate[k] * W_down[k * D_out + j];
        }
        out_row[j] = acc;
    }
}


/**
 * Fused LayerNorm + FFN: LN(x) → W_up + b_up → GELU → W_down + b_down
 * Eliminates one additional dispatch by fusing LayerNorm into the FFN.
 */
kernel void fused_ln_ffn(
    device const float* x       [[buffer(0)]],   // (rows, D)
    device const float* ln_w    [[buffer(1)]],   // (D,) LayerNorm weight
    device const float* ln_b    [[buffer(2)]],   // (D,) LayerNorm bias
    device const float* W_up    [[buffer(3)]],   // (D, D_mid)
    device const float* b_up    [[buffer(4)]],   // (D_mid,)
    device const float* W_down  [[buffer(5)]],   // (D_mid, D_out)
    device const float* b_down  [[buffer(6)]],   // (D_out,)
    device float*       out     [[buffer(7)]],   // (rows, D_out)
    constant uint&      D_in    [[buffer(8)]],
    constant uint&      D_mid   [[buffer(9)]],
    constant uint&      D_out   [[buffer(10)]],
    constant float&     ln_eps  [[buffer(11)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  tg_size   [[threads_per_threadgroup]]
)
{
    const uint row = tg_id;
    device const float* x_row = x + row * D_in;

    // Phase 0: LayerNorm — compute mean and variance
    // Use shared memory for parallel reduction
    threadgroup float shared_sum[256];
    threadgroup float shared_sq_sum[256];
    threadgroup float ln_result[1280];  // Normalized input

    // Compute partial sums
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (uint k = tid; k < D_in; k += tg_size) {
        float val = x_row[k];
        local_sum += val;
        local_sq_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(D_in);
    float var = shared_sq_sum[0] / float(D_in) - mean * mean;
    float inv_std = rsqrt(var + ln_eps);

    // Apply LayerNorm: (x - mean) * inv_std * weight + bias
    for (uint k = tid; k < D_in; k += tg_size) {
        ln_result[k] = (x_row[k] - mean) * inv_std * ln_w[k] + ln_b[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: intermediate = GELU(ln_result @ W_up + b_up)
    threadgroup float intermediate[5120];

    for (uint j = tid; j < D_mid; j += tg_size) {
        float acc = b_up[j];
        for (uint k = 0; k < D_in; k++) {
            acc += ln_result[k] * W_up[k * D_mid + j];
        }
        intermediate[j] = gelu_tanh(acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: output = intermediate @ W_down + b_down
    device float* out_row = out + row * D_out;
    for (uint j = tid; j < D_out; j += tg_size) {
        float acc = b_down[j];
        for (uint k = 0; k < D_mid; k++) {
            acc += intermediate[k] * W_down[k * D_out + j];
        }
        out_row[j] = acc;
    }
}
