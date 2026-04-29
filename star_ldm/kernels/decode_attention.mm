/**
 * Decode-N Attention — Objective-C++ dispatch for PyTorch MPS.
 *
 * Provides: decode_attention(Q, K, V, scale) -> O
 *
 * Dispatches the Metal compute shader for the "few queries, moderate KV"
 * attention pattern. One threadgroup per (batch, head), 256 threads per
 * threadgroup (8 queries × 32 threads per query).
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Cached pipeline states
static id<MTLComputePipelineState> pipeline_f32 = nil;
static id<MTLComputePipelineState> pipeline_f16 = nil;
static id<MTLLibrary> library = nil;


static void ensure_pipeline() {
    if (library != nil) return;

    @autoreleasepool {
        NSError* error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Load shader source from file next to this .mm
        NSString* path = [NSString stringWithUTF8String:__FILE__];
        NSString* dir = [path stringByDeletingLastPathComponent];
        NSString* metalPath = [dir stringByAppendingPathComponent:@"decode_attention.metal"];
        NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        TORCH_CHECK(error == nil, "Failed to read decode_attention.metal: ",
                    [[error localizedDescription] UTF8String]);

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion3_0;

        library = [device newLibraryWithSource:source options:options error:&error];
        TORCH_CHECK(error == nil, "Metal compilation error: ",
                    [[error localizedDescription] UTF8String]);

        // Create pipeline for f32
        id<MTLFunction> fn_f32 = [library newFunctionWithName:@"decode_n_attention"];
        TORCH_CHECK(fn_f32 != nil, "decode_n_attention function not found");
        pipeline_f32 = [device newComputePipelineStateWithFunction:fn_f32 error:&error];
        TORCH_CHECK(error == nil, "Pipeline creation error (f32): ",
                    [[error localizedDescription] UTF8String]);

        // Create pipeline for f16
        id<MTLFunction> fn_f16 = [library newFunctionWithName:@"decode_n_attention_f16"];
        TORCH_CHECK(fn_f16 != nil, "decode_n_attention_f16 function not found");
        pipeline_f16 = [device newComputePipelineStateWithFunction:fn_f16 error:&error];
        TORCH_CHECK(error == nil, "Pipeline creation error (f16): ",
                    [[error localizedDescription] UTF8String]);
    }
}


/**
 * decode_attention(Q, K, V, scale) -> O
 *
 * Q: (B, H, N_Q, D_HEAD)  — query tokens
 * K: (B, H, S_KV, D_HEAD) — key cache (prefix + new)
 * V: (B, H, S_KV, D_HEAD) — value cache
 * scale: float             — 1/sqrt(d_head)
 *
 * Returns: O (B, H, N_Q, D_HEAD)
 */
torch::Tensor decode_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    double scale
) {
    TORCH_CHECK(Q.is_mps(), "Q must be on MPS device");
    TORCH_CHECK(K.is_mps(), "K must be on MPS device");
    TORCH_CHECK(V.is_mps(), "V must be on MPS device");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D (B, H, N, D)");
    TORCH_CHECK(K.dim() == 4, "K must be 4D (B, H, S, D)");

    const uint32_t B      = Q.size(0);
    const uint32_t H      = Q.size(1);
    const uint32_t N_Q    = Q.size(2);
    const uint32_t D_HEAD = Q.size(3);
    const uint32_t S_KV   = K.size(2);
    const float scale_f   = static_cast<float>(scale);

    TORCH_CHECK(N_Q <= 16, "decode_attention supports N_Q <= 16, got ", N_Q);
    TORCH_CHECK(D_HEAD <= 128, "decode_attention supports D_HEAD <= 128, got ", D_HEAD);
    TORCH_CHECK(D_HEAD % 32 == 0, "D_HEAD must be divisible by 32, got ", D_HEAD);

    // Output tensor
    auto O = torch::empty_like(Q);

    ensure_pipeline();

    bool use_f16 = (Q.scalar_type() == torch::kHalf);
    id<MTLComputePipelineState> pipeline = use_f16 ? pipeline_f16 : pipeline_f32;

    // Use PyTorch's MPS stream for proper synchronization
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];

            [enc setBuffer:at::native::mps::getMTLBufferStorage(Q) offset:Q.storage_offset() * Q.element_size() atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(K) offset:K.storage_offset() * K.element_size() atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(V) offset:V.storage_offset() * V.element_size() atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(O) offset:O.storage_offset() * O.element_size() atIndex:3];

            [enc setBytes:&N_Q     length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&S_KV    length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&D_HEAD  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&H       length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&scale_f length:sizeof(float)    atIndex:8];

            // One threadgroup per (batch, head), N_Q * 32 threads per threadgroup
            uint32_t threads_per_q = 32;
            uint32_t tg_size = N_Q * threads_per_q;
            uint32_t total_threads = B * H * tg_size;

            MTLSize gridSize = MTLSizeMake(total_threads, 1, 1);
            MTLSize tgSize   = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attention", &decode_attention,
          "Decode-N attention (Metal kernel): few queries attending to KV cache",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));
}
