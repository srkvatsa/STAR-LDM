/**
 * Objective-C++ dispatch for fused 8-token attention Metal kernel.
 *
 * Uses PyTorch 2.x MPS C++ API: getCurrentMPSStream(), getMTLBufferStorage().
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLLibrary> _library = nil;
static id<MTLComputePipelineState> _pipeline = nil;

static id<MTLComputePipelineState> get_pipeline() {
    if (_pipeline != nil) return _pipeline;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString* path = [NSString stringWithUTF8String:__FILE__];
    NSString* dir = [path stringByDeletingLastPathComponent];
    NSString* metalPath = [dir stringByAppendingPathComponent:@"tiny_attention.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read tiny_attention.metal");

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile tiny_attention.metal: ",
                [[error localizedDescription] UTF8String]);

    id<MTLFunction> func = [_library newFunctionWithName:@"tiny_attention_kernel"];
    TORCH_CHECK(func != nil, "tiny_attention_kernel not found");

    _pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create tiny_attention pipeline");

    return _pipeline;
}


torch::Tensor tiny_attention_metal(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor q_gamma,
    torch::Tensor k_gamma,
    double dim_head_scale,
    double attn_scale
) {
    TORCH_CHECK(Q.is_mps() && Q.dtype() == torch::kFloat32, "Q must be float32 on MPS");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D_head = Q.size(3);

    TORCH_CHECK(S == 8, "tiny_attention_metal requires seq_len=8, got ", S);

    float dim_head_scale_f = (float)dim_head_scale;
    float attn_scale_f = (float)attn_scale;

    auto out = torch::empty_like(Q);

    id<MTLComputePipelineState> pipeline = get_pipeline();

    NSUInteger tg_size = 128;
    NSUInteger shared_mem = (2 * S * D_head + tg_size) * sizeof(float);
    uint32_t num_tgs = B * H;

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(Q)       offset:Q.storage_offset() * Q.element_size()       atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(K)       offset:K.storage_offset() * K.element_size()       atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(V)       offset:V.storage_offset() * V.element_size()       atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(q_gamma) offset:q_gamma.storage_offset() * q_gamma.element_size() atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(k_gamma) offset:k_gamma.storage_offset() * k_gamma.element_size() atIndex:4];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)     offset:out.storage_offset() * out.element_size()   atIndex:5];
            [enc setBytes:&D_head length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&dim_head_scale_f length:sizeof(float) atIndex:7];
            [enc setBytes:&attn_scale_f length:sizeof(float) atIndex:8];
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

            MTLSize grid = MTLSizeMake(num_tgs, 1, 1);
            MTLSize tg = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiny_attention", &tiny_attention_metal,
          "Fused 8-token attention with QK-norm (Metal)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("q_gamma"), py::arg("k_gamma"),
          py::arg("dim_head_scale"), py::arg("attn_scale"));
}
