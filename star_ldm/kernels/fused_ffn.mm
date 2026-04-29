/**
 * Fused FFN — Objective-C++ dispatch for PyTorch MPS.
 *
 * Provides:
 *   fused_ffn(x, W_up, b_up, W_down, b_down) -> out
 *   fused_ln_ffn(x, ln_w, ln_b, W_up, b_up, W_down, b_down, eps) -> out
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLComputePipelineState> pipeline_ffn = nil;
static id<MTLComputePipelineState> pipeline_ln_ffn = nil;
static id<MTLLibrary> library = nil;


static void ensure_pipelines() {
    if (library != nil) return;

    @autoreleasepool {
        NSError* error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        NSString* path = [NSString stringWithUTF8String:__FILE__];
        NSString* dir = [path stringByDeletingLastPathComponent];
        NSString* metalPath = [dir stringByAppendingPathComponent:@"fused_ffn.metal"];
        NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        TORCH_CHECK(error == nil, "Failed to read fused_ffn.metal: ",
                    [[error localizedDescription] UTF8String]);

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion3_0;

        library = [device newLibraryWithSource:source options:options error:&error];
        TORCH_CHECK(error == nil, "Metal compilation error: ",
                    [[error localizedDescription] UTF8String]);

        id<MTLFunction> fn_ffn = [library newFunctionWithName:@"fused_ffn"];
        TORCH_CHECK(fn_ffn != nil, "fused_ffn function not found");
        pipeline_ffn = [device newComputePipelineStateWithFunction:fn_ffn error:&error];
        TORCH_CHECK(error == nil, "Pipeline error (fused_ffn)");

        id<MTLFunction> fn_ln_ffn = [library newFunctionWithName:@"fused_ln_ffn"];
        TORCH_CHECK(fn_ln_ffn != nil, "fused_ln_ffn function not found");
        pipeline_ln_ffn = [device newComputePipelineStateWithFunction:fn_ln_ffn error:&error];
        TORCH_CHECK(error == nil, "Pipeline error (fused_ln_ffn)");
    }
}


torch::Tensor fused_ffn(
    const torch::Tensor& x,
    const torch::Tensor& W_up,
    const torch::Tensor& b_up,
    const torch::Tensor& W_down,
    const torch::Tensor& b_down
) {
    TORCH_CHECK(x.is_mps(), "x must be on MPS");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, D_in)");

    const uint32_t rows  = x.size(0);
    const uint32_t D_in  = x.size(1);
    const uint32_t D_mid = W_up.size(1);
    const uint32_t D_out = W_down.size(1);

    auto out = torch::empty({(int64_t)rows, (int64_t)D_out}, x.options());

    ensure_pipelines();

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pipeline_ffn];

            [enc setBuffer:at::native::mps::getMTLBufferStorage(x)      offset:x.storage_offset() * x.element_size()           atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(W_up)   offset:W_up.storage_offset() * W_up.element_size()     atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(b_up)   offset:b_up.storage_offset() * b_up.element_size()     atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(W_down) offset:W_down.storage_offset() * W_down.element_size() atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(b_down) offset:b_down.storage_offset() * b_down.element_size() atIndex:4];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)    offset:out.storage_offset() * out.element_size()       atIndex:5];
            [enc setBytes:&D_in  length:sizeof(uint32_t) atIndex:6];
            [enc setBytes:&D_mid length:sizeof(uint32_t) atIndex:7];
            [enc setBytes:&D_out length:sizeof(uint32_t) atIndex:8];

            // One threadgroup per row, 256 threads per threadgroup
            uint32_t tg_size = 256;
            MTLSize gridSize = MTLSizeMake(rows * tg_size, 1, 1);
            MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return out;
}


torch::Tensor fused_ln_ffn(
    const torch::Tensor& x,
    const torch::Tensor& ln_w,
    const torch::Tensor& ln_b,
    const torch::Tensor& W_up,
    const torch::Tensor& b_up,
    const torch::Tensor& W_down,
    const torch::Tensor& b_down,
    double eps
) {
    TORCH_CHECK(x.is_mps(), "x must be on MPS");
    TORCH_CHECK(x.dim() == 2, "x must be 2D (rows, D)");

    const uint32_t rows  = x.size(0);
    const uint32_t D_in  = x.size(1);
    const uint32_t D_mid = W_up.size(1);
    const uint32_t D_out = W_down.size(1);
    const float eps_f = static_cast<float>(eps);

    auto out = torch::empty({(int64_t)rows, (int64_t)D_out}, x.options());

    ensure_pipelines();

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pipeline_ln_ffn];

            [enc setBuffer:at::native::mps::getMTLBufferStorage(x)      offset:x.storage_offset() * x.element_size()           atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(ln_w)   offset:ln_w.storage_offset() * ln_w.element_size()     atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(ln_b)   offset:ln_b.storage_offset() * ln_b.element_size()     atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(W_up)   offset:W_up.storage_offset() * W_up.element_size()     atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(b_up)   offset:b_up.storage_offset() * b_up.element_size()     atIndex:4];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(W_down) offset:W_down.storage_offset() * W_down.element_size() atIndex:5];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(b_down) offset:b_down.storage_offset() * b_down.element_size() atIndex:6];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)    offset:out.storage_offset() * out.element_size()       atIndex:7];
            [enc setBytes:&D_in  length:sizeof(uint32_t) atIndex:8];
            [enc setBytes:&D_mid length:sizeof(uint32_t) atIndex:9];
            [enc setBytes:&D_out length:sizeof(uint32_t) atIndex:10];
            [enc setBytes:&eps_f length:sizeof(float)    atIndex:11];

            uint32_t tg_size = 256;
            MTLSize gridSize = MTLSizeMake(rows * tg_size, 1, 1);
            MTLSize tgSize = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ffn", &fused_ffn,
          "Fused FFN: Linear + GELU + Linear (Metal)",
          py::arg("x"), py::arg("W_up"), py::arg("b_up"),
          py::arg("W_down"), py::arg("b_down"));
    m.def("fused_ln_ffn", &fused_ln_ffn,
          "Fused LayerNorm + FFN (Metal)",
          py::arg("x"), py::arg("ln_w"), py::arg("ln_b"),
          py::arg("W_up"), py::arg("b_up"),
          py::arg("W_down"), py::arg("b_down"), py::arg("eps"));
}
