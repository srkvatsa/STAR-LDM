/**
 * Fused speculative decoding verification — Metal compute shader.
 *
 * Single dispatch fuses the verification step for K candidate tokens:
 *   1. Softmax over vocab (V=50257) for both draft and target logits
 *   2. Compute acceptance ratio: min(1, p_target/p_draft) for each position
 *   3. Compare against uniform random numbers
 *   4. Find first rejection index (parallel scan)
 *   5. If rejected: compute adjusted distribution (p_target - p_draft)+
 *
 * Shape: draft_logits and target_logits are (K, V) where K=4, V=50257
 * Output: accepted (K,) bool mask, adjusted_probs (V,) for rejected position
 *
 * Threadgroup: one per candidate position (K threadgroups).
 * Threads cooperate on the vocab-dimension softmax reduction.
 */

#include <metal_stdlib>
using namespace metal;


// --- Kernel 1: Compute softmax probabilities for draft and target ---
// Writes p_draft (K, V) and p_target (K, V)
kernel void softmax_logits_kernel(
    device const float* draft_logits   [[buffer(0)]],  // (K, V)
    device const float* target_logits  [[buffer(1)]],  // (K, V)
    device float*       p_draft        [[buffer(2)]],  // (K, V)
    device float*       p_target       [[buffer(3)]],  // (K, V)
    constant uint&      V              [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],       // k index
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    threadgroup float* shared          [[threadgroup_binding(0)]]
) {
    uint k = tgid;
    uint base = k * V;

    // --- Draft softmax ---
    // Pass 1: find max (for numerical stability)
    float local_max = -INFINITY;
    for (uint v = tid; v < V; v += tg_sz) {
        local_max = max(local_max, draft_logits[base + v]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float draft_max = shared[0];

    // Pass 2: exp and sum
    float local_sum = 0.0f;
    for (uint v = tid; v < V; v += tg_sz) {
        float e = exp(draft_logits[base + v] - draft_max);
        p_draft[base + v] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float draft_sum = shared[0];

    // Pass 3: normalize
    float inv_draft_sum = 1.0f / draft_sum;
    for (uint v = tid; v < V; v += tg_sz) {
        p_draft[base + v] *= inv_draft_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Target softmax ---
    local_max = -INFINITY;
    for (uint v = tid; v < V; v += tg_sz) {
        local_max = max(local_max, target_logits[base + v]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float target_max = shared[0];

    local_sum = 0.0f;
    for (uint v = tid; v < V; v += tg_sz) {
        float e = exp(target_logits[base + v] - target_max);
        p_target[base + v] = e;
        local_sum += e;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_sz / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float target_sum = shared[0];

    float inv_target_sum = 1.0f / target_sum;
    for (uint v = tid; v < V; v += tg_sz) {
        p_target[base + v] *= inv_target_sum;
    }
}


// --- Kernel 2: Accept/reject and compute adjusted distribution ---
// For each position k, check acceptance. On first rejection, compute
// adjusted distribution p_adjusted = max(0, p_target - p_draft) / Z
kernel void verify_and_adjust_kernel(
    device const float*  p_draft        [[buffer(0)]],  // (K, V)
    device const float*  p_target       [[buffer(1)]],  // (K, V)
    device const int*    draft_tokens   [[buffer(2)]],  // (K,) sampled token ids
    device const float*  rand_uniform   [[buffer(3)]],  // (K,) uniform random [0,1)
    device int*          first_reject   [[buffer(4)]],  // scalar output
    device float*        adjusted_probs [[buffer(5)]],  // (V,) output adjusted distribution
    constant uint&       K              [[buffer(6)]],
    constant uint&       V              [[buffer(7)]],
    uint tid    [[thread_position_in_grid]]
) {
    // Single thread executes this (K is small, typically 4)
    if (tid != 0) return;

    int reject_idx = (int)K;  // default: all accepted

    for (uint k = 0; k < K; k++) {
        int token = draft_tokens[k];
        float pd = p_draft[k * V + token];
        float pt = p_target[k * V + token];

        // Acceptance probability: min(1, pt / pd)
        float accept_prob = (pd > 0.0f) ? min(1.0f, pt / pd) : 0.0f;

        if (rand_uniform[k] >= accept_prob) {
            reject_idx = (int)k;
            break;
        }
    }

    first_reject[0] = reject_idx;

    // If there's a rejection, compute adjusted distribution
    if ((uint)reject_idx < K) {
        uint k = (uint)reject_idx;
        // adjusted = max(0, p_target[k] - p_draft[k]) for each vocab element
        float sum = 0.0f;
        for (uint v = 0; v < V; v++) {
            float adj = max(0.0f, p_target[k * V + v] - p_draft[k * V + v]);
            adjusted_probs[v] = adj;
            sum += adj;
        }
        // Normalize
        if (sum > 0.0f) {
            float inv_sum = 1.0f / sum;
            for (uint v = 0; v < V; v++) {
                adjusted_probs[v] *= inv_sum;
            }
        }
    }
}
