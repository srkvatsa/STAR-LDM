/**
 * Objective-C++ dispatch file for the fused DDPM step Metal kernel.
 *
 * Uses PyTorch 2.x MPS C++ API: getCurrentMPSStream(), getMTLBufferStorage().
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Cache the compiled pipeline state
static id<MTLComputePipelineState> _pipeline = nil;
static id<MTLLibrary> _library = nil;

static id<MTLComputePipelineState> get_pipeline() {
    if (_pipeline != nil) return _pipeline;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    NSString* path = [NSString stringWithUTF8String:__FILE__];
    NSString* dir = [path stringByDeletingLastPathComponent];
    NSString* metalPath = [dir stringByAppendingPathComponent:@"ddpm_step.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read ddpm_step.metal: ",
                [[error localizedDescription] UTF8String]);

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile ddpm_step.metal: ",
                [[error localizedDescription] UTF8String]);

    id<MTLFunction> func = [_library newFunctionWithName:@"ddpm_step_kernel"];
    TORCH_CHECK(func != nil, "ddpm_step_kernel function not found in metal source");

    _pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create pipeline state: ",
                [[error localizedDescription] UTF8String]);

    return _pipeline;
}


torch::Tensor ddpm_step_metal(
    torch::Tensor z_t,
    torch::Tensor eps,
    torch::Tensor noise,
    torch::Tensor alpha2,
    torch::Tensor alpha2_next,
    double var_lambda
) {
    TORCH_CHECK(z_t.is_mps(), "z_t must be on MPS device");
    TORCH_CHECK(z_t.dtype() == torch::kFloat32, "z_t must be float32");
    TORCH_CHECK(z_t.is_contiguous(), "z_t must be contiguous");

    uint32_t B = z_t.size(0);
    uint32_t D = z_t.size(1);
    uint32_t total = B * D;

    auto alpha2_flat = alpha2.contiguous().view({B});
    auto alpha2_next_flat = alpha2_next.contiguous().view({B});
    auto var_lambda_t = torch::tensor({(float)var_lambda},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    auto z_out = torch::empty_like(z_t);

    id<MTLComputePipelineState> pipeline = get_pipeline();

    // Use PyTorch's MPS stream for proper synchronization
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(z_t)            offset:z_t.storage_offset() * z_t.element_size()            atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(eps)             offset:eps.storage_offset() * eps.element_size()             atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(noise)           offset:noise.storage_offset() * noise.element_size()         atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2_flat)     offset:alpha2_flat.storage_offset() * alpha2_flat.element_size()     atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(alpha2_next_flat) offset:alpha2_next_flat.storage_offset() * alpha2_next_flat.element_size() atIndex:4];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(var_lambda_t)    offset:var_lambda_t.storage_offset() * var_lambda_t.element_size()    atIndex:5];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(z_out)           offset:z_out.storage_offset() * z_out.element_size()         atIndex:6];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:7];

            MTLSize gridSize = MTLSizeMake(total, 1, 1);
            NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)total);
            MTLSize tgSize = MTLSizeMake(threadGroupSize, 1, 1);
            [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        }
    });

    return z_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ddpm_step", &ddpm_step_metal,
          "Fused DDPM denoising step (Metal compute shader)",
          py::arg("z_t"), py::arg("eps"), py::arg("noise"),
          py::arg("alpha2"), py::arg("alpha2_next"), py::arg("var_lambda"));
}
