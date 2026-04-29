/**
 * Fused 8-token attention — Metal compute shader.
 *
 * Custom attention for seq_len=8. The 8x8 attention matrix per head fits
 * entirely in registers. Single dispatch fuses:
 *   1. QK-norm (L2 normalize Q, K with shared memory reduction)
 *   2. QK^T matmul (8x8, fully unrolled in registers)
 *   3. Softmax (all 8 values fit in registers)
 *   4. Attention x V (8 x D_head output)
 *
 * One threadgroup per (batch, head) pair.
 * Threads cooperate on the D_head dimension.
 *
 * Tensor layout: Q, K, V are (B, H, 8, D_head) contiguous float
 * where H=8, D_head=96
 */

#include <metal_stdlib>
using namespace metal;

constant uint SEQ_LEN = 8;

/**
 * Helper: parallel L2 norm reduction across D_head for one (b, h, s) vector.
 * Returns the L2 norm (shared across all threads in the threadgroup via shared mem).
 *
 * Each thread handles a subset of the D_head dimensions.
 */

kernel void tiny_attention_kernel(
    device const float* Q           [[buffer(0)]],   // (B, H, 8, D_head)
    device const float* K           [[buffer(1)]],
    device const float* V           [[buffer(2)]],
    device const float* q_gamma     [[buffer(3)]],   // (D_head,)
    device const float* k_gamma     [[buffer(4)]],   // (D_head,)
    device float*       out         [[buffer(5)]],   // (B, H, 8, D_head)
    constant uint&      D_head      [[buffer(6)]],
    constant float&     dim_head_scale [[buffer(7)]], // sqrt(D_head)
    constant float&     attn_scale  [[buffer(8)]],   // 1/sqrt(D_head)
    uint tgid   [[threadgroup_position_in_grid]],     // b * H + h
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared       [[threadgroup_binding(0)]]
) {
    // Each threadgroup handles one (batch, head) pair
    // tgid = b * H + h
    uint bh_offset = tgid * SEQ_LEN * D_head;  // offset into (B, H, 8, D_head)

    // --- Step 1: QK-norm ---
    // Normalize Q and K vectors for all 8 positions.
    // We use shared memory to store the normalized Q and K.
    // shared layout: [Q_normed: 8*D_head] [K_normed: 8*D_head] [scratch: tg_sz]
    threadgroup float* Q_norm = shared;                          // 8 * D_head
    threadgroup float* K_norm = shared + SEQ_LEN * D_head;       // 8 * D_head
    threadgroup float* scratch = shared + 2 * SEQ_LEN * D_head;  // tg_sz

    // Normalize each of the 8 Q vectors
    for (uint s = 0; s < SEQ_LEN; s++) {
        uint vec_offset = bh_offset + s * D_head;

        // Parallel reduction for L2 norm
        float partial = 0.0f;
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = Q[vec_offset + d];
            partial += val * val;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float q_norm_val = max(sqrt(scratch[0]), 1e-8f);

        // Write normalized Q
        for (uint d = tid; d < D_head; d += tg_sz) {
            Q_norm[s * D_head + d] = Q[vec_offset + d] / q_norm_val * dim_head_scale * q_gamma[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize each of the 8 K vectors
    for (uint s = 0; s < SEQ_LEN; s++) {
        uint vec_offset = bh_offset + s * D_head;

        float partial = 0.0f;
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = K[vec_offset + d];
            partial += val * val;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_sz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float k_norm_val = max(sqrt(scratch[0]), 1e-8f);

        for (uint d = tid; d < D_head; d += tg_sz) {
            K_norm[s * D_head + d] = K[vec_offset + d] / k_norm_val * dim_head_scale * k_gamma[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Step 2: QK^T matmul (8x8) ---
    // Each thread computes a subset of the 64 dot products.
    // We store the 8x8 attention matrix in shared memory.
    threadgroup float* attn_matrix = scratch;  // reuse scratch, need 64 floats

    // Each element attn[i][j] = sum_d Q_norm[i][d] * K_norm[j][d] * attn_scale
    for (uint idx = tid; idx < SEQ_LEN * SEQ_LEN; idx += tg_sz) {
        uint i = idx / SEQ_LEN;
        uint j = idx % SEQ_LEN;
        float dot = 0.0f;
        for (uint d = 0; d < D_head; d++) {
            dot += Q_norm[i * D_head + d] * K_norm[j * D_head + d];
        }
        attn_matrix[idx] = dot * attn_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 3: Softmax (row-wise, 8 values per row) ---
    // Each thread handles one or more rows
    for (uint i = tid; i < SEQ_LEN; i += tg_sz) {
        // Find max for numerical stability
        float max_val = attn_matrix[i * SEQ_LEN];
        for (uint j = 1; j < SEQ_LEN; j++) {
            max_val = max(max_val, attn_matrix[i * SEQ_LEN + j]);
        }
        // Exp and sum
        float sum_exp = 0.0f;
        for (uint j = 0; j < SEQ_LEN; j++) {
            float e = exp(attn_matrix[i * SEQ_LEN + j] - max_val);
            attn_matrix[i * SEQ_LEN + j] = e;
            sum_exp += e;
        }
        // Normalize
        float inv_sum = 1.0f / sum_exp;
        for (uint j = 0; j < SEQ_LEN; j++) {
            attn_matrix[i * SEQ_LEN + j] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 4: Attention x V ---
    // out[i][d] = sum_j attn[i][j] * V[j][d]
    for (uint i = 0; i < SEQ_LEN; i++) {
        for (uint d = tid; d < D_head; d += tg_sz) {
            float val = 0.0f;
            for (uint j = 0; j < SEQ_LEN; j++) {
                val += attn_matrix[i * SEQ_LEN + j] * V[bh_offset + j * D_head + d];
            }
            out[bh_offset + i * D_head + d] = val;
        }
    }
}
