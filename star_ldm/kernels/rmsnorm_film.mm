/**
 * Objective-C++ dispatch for fused RMSNorm + FiLM Metal kernels.
 *
 * Uses PyTorch 2.x MPS C++ API: getCurrentMPSStream(), getMTLBufferStorage().
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLLibrary> _library = nil;
static id<MTLComputePipelineState> _film_pipeline = nil;
static id<MTLComputePipelineState> _norm_pipeline = nil;

static void ensure_library() {
    if (_library != nil) return;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString* path = [NSString stringWithUTF8String:__FILE__];
    NSString* dir = [path stringByDeletingLastPathComponent];
    NSString* metalPath = [dir stringByAppendingPathComponent:@"rmsnorm_film.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read rmsnorm_film.metal");

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;

    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile rmsnorm_film.metal: ",
                [[error localizedDescription] UTF8String]);
}

static id<MTLComputePipelineState> get_film_pipeline() {
    if (_film_pipeline != nil) return _film_pipeline;
    ensure_library();
    NSError* error = nil;
    id<MTLFunction> func = [_library newFunctionWithName:@"rmsnorm_film_kernel"];
    TORCH_CHECK(func != nil, "rmsnorm_film_kernel not found");
    _film_pipeline = [MTLCreateSystemDefaultDevice() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create rmsnorm_film pipeline");
    return _film_pipeline;
}

static id<MTLComputePipelineState> get_norm_pipeline() {
    if (_norm_pipeline != nil) return _norm_pipeline;
    ensure_library();
    NSError* error = nil;
    id<MTLFunction> func = [_library newFunctionWithName:@"rmsnorm_kernel"];
    TORCH_CHECK(func != nil, "rmsnorm_kernel not found");
    _norm_pipeline = [MTLCreateSystemDefaultDevice() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create rmsnorm pipeline");
    return _norm_pipeline;
}


torch::Tensor rmsnorm_film_metal(
    torch::Tensor x,
    torch::Tensor gamma,
    double dim_scale,
    torch::Tensor film_scale,
    torch::Tensor film_shift
) {
    TORCH_CHECK(x.is_mps() && x.dtype() == torch::kFloat32, "x must be float32 on MPS");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t num_rows = x.numel() / x.size(-1);
    uint32_t D = (uint32_t)x.size(-1);
    float dim_scale_f = (float)dim_scale;
    uint32_t film_stride_val = 1;

    auto film_s = film_scale.expand_as(x).contiguous();
    auto film_sh = film_shift.expand_as(x).contiguous();
    auto out = torch::empty_like(x);

    id<MTLComputePipelineState> pipeline = get_film_pipeline();
    NSUInteger tg_size = 256;
    NSUInteger shared_mem = tg_size * sizeof(float);

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(x)        offset:x.storage_offset() * x.element_size()        atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(gamma)    offset:gamma.storage_offset() * gamma.element_size() atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(film_s)   offset:film_s.storage_offset() * film_s.element_size()   atIndex:2];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(film_sh)  offset:film_sh.storage_offset() * film_sh.element_size() atIndex:3];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)      offset:out.storage_offset() * out.element_size()     atIndex:4];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:5];
            [enc setBytes:&dim_scale_f length:sizeof(float) atIndex:6];
            [enc setBytes:&film_stride_val length:sizeof(uint32_t) atIndex:7];
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

            MTLSize grid = MTLSizeMake((NSUInteger)num_rows, 1, 1);
            MTLSize tg = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
    });

    return out;
}


torch::Tensor rmsnorm_metal(
    torch::Tensor x,
    torch::Tensor gamma,
    double dim_scale
) {
    TORCH_CHECK(x.is_mps() && x.dtype() == torch::kFloat32, "x must be float32 on MPS");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t num_rows = x.numel() / x.size(-1);
    uint32_t D = (uint32_t)x.size(-1);
    float dim_scale_f = (float)dim_scale;
    auto out = torch::empty_like(x);

    id<MTLComputePipelineState> pipeline = get_norm_pipeline();
    NSUInteger tg_size = 256;
    NSUInteger shared_mem = tg_size * sizeof(float);

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

            [enc setComputePipelineState:pipeline];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(x)     offset:x.storage_offset() * x.element_size()     atIndex:0];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(gamma) offset:gamma.storage_offset() * gamma.element_size() atIndex:1];
            [enc setBuffer:at::native::mps::getMTLBufferStorage(out)   offset:out.storage_offset() * out.element_size()  atIndex:2];
            [enc setBytes:&D length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&dim_scale_f length:sizeof(float) atIndex:4];
            [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

            MTLSize grid = MTLSizeMake((NSUInteger)num_rows, 1, 1);
            MTLSize tg = MTLSizeMake(tg_size, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
    });

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_film", &rmsnorm_film_metal,
          "Fused RMSNorm + FiLM conditioning (Metal)",
          py::arg("x"), py::arg("gamma"), py::arg("dim_scale"),
          py::arg("film_scale"), py::arg("film_shift"));
    m.def("rmsnorm", &rmsnorm_metal,
          "RMSNorm without FiLM (Metal)",
          py::arg("x"), py::arg("gamma"), py::arg("dim_scale"));
}
