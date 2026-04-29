/**
 * Fused DDPM denoising step — Metal compute shader.
 *
 * Single dispatch computes the full DDPM update for (B, D) tensors:
 *   alpha2_now = alpha2 / alpha2_next
 *   min_var = exp(log(1.0f +-alpha2_next) - log(1.0f +-alpha2)) * (1 - alpha2_now)
 *   max_var = 1 - alpha2_now
 *   sigma = exp(var_lambda * log(max_var) + (1-var_lambda) * log(min_var))
 *   z_next = 1/sqrt(alpha2_now) * (z_t - (1-alpha2_now)/sqrt(1-alpha2) * eps)
 *            + sqrt(sigma) * noise
 *
 * One thread per element. All intermediate values stay in registers.
 *
 * Tensor layout:
 *   z_t, eps, noise: (B, D) contiguous float
 *   alpha2, alpha2_next: (B, 1) broadcast over D
 */

#include <metal_stdlib>
using namespace metal;

kernel void ddpm_step_kernel(
    device const float* z_t        [[buffer(0)]],
    device const float* eps        [[buffer(1)]],
    device const float* noise      [[buffer(2)]],
    device const float* alpha2     [[buffer(3)]],  // (B, 1)
    device const float* alpha2_next [[buffer(4)]], // (B, 1)
    device const float* var_lambda_buf [[buffer(5)]], // scalar
    device float*       z_out      [[buffer(6)]],
    constant uint&      D          [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint b = tid / D;
    uint d = tid % D;

    float a2      = alpha2[b];
    float a2_next = alpha2_next[b];
    float vl      = var_lambda_buf[0];

    // alpha2_now = alpha2 / alpha2_next
    float a2_now = a2 / a2_next;

    // Variance bounds
    float min_var = exp(log(max(1.0f - a2_next, 1e-8f)) - log(max(1.0f - a2, 1e-8f))) * (1.0f - a2_now);
    float max_var = 1.0f - a2_now;

    // Clamp to avoid log(0)
    min_var = max(min_var, 1e-8f);
    max_var = max(max_var, 1e-8f);

    // Interpolated variance in log-space
    float sigma = exp(vl * log(max_var) + (1.0f - vl) * log(min_var));

    // Mean computation
    float inv_sqrt_a2_now = rsqrt(a2_now);
    float coeff = (1.0f - a2_now) * rsqrt(1.0f - a2);

    float z = z_t[tid];
    float e = eps[tid];
    float n = noise[tid];

    z_out[tid] = inv_sqrt_a2_now * (z - coeff * e) + sqrt(sigma) * n;
}
