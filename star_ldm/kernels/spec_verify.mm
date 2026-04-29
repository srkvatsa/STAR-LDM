/**
 * Objective-C++ dispatch for fused speculative verification Metal kernels.
 *
 * Uses PyTorch 2.x MPS C++ API: getCurrentMPSStream(), getMTLBufferStorage().
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLLibrary> _library = nil;
static id<MTLComputePipelineState> _softmax_pipeline = nil;
static id<MTLComputePipelineState> _verify_pipeline = nil;

static void ensure_library() {
    if (_library != nil) return;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString* path = [NSString stringWithUTF8String:__FILE__];
    NSString* dir = [path stringByDeletingLastPathComponent];
    NSString* metalPath = [dir stringByAppendingPathComponent:@"spec_verify.metal"];

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:metalPath
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    TORCH_CHECK(error == nil, "Failed to read spec_verify.metal");

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    _library = [device newLibraryWithSource:source options:opts error:&error];
    TORCH_CHECK(error == nil, "Failed to compile spec_verify.metal: ",
                [[error localizedDescription] UTF8String]);
}

static id<MTLComputePipelineState> get_softmax_pipeline() {
    if (_softmax_pipeline != nil) return _softmax_pipeline;
    ensure_library();
    NSError* error = nil;
    id<MTLFunction> func = [_library newFunctionWithName:@"softmax_logits_kernel"];
    TORCH_CHECK(func != nil, "softmax_logits_kernel not found");
    _softmax_pipeline = [MTLCreateSystemDefaultDevice() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create softmax pipeline");
    return _softmax_pipeline;
}

static id<MTLComputePipelineState> get_verify_pipeline() {
    if (_verify_pipeline != nil) return _verify_pipeline;
    ensure_library();
    NSError* error = nil;
    id<MTLFunction> func = [_library newFunctionWithName:@"verify_and_adjust_kernel"];
    TORCH_CHECK(func != nil, "verify_and_adjust_kernel not found");
    _verify_pipeline = [MTLCreateSystemDefaultDevice() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(error == nil, "Failed to create verify pipeline");
    return _verify_pipeline;
}


std::tuple<torch::Tensor, torch::Tensor> speculative_verify_metal(
    torch::Tensor draft_logits,
    torch::Tensor target_logits,
    torch::Tensor draft_tokens,
    torch::Tensor rand_uniform
) {
    TORCH_CHECK(draft_logits.is_mps(), "draft_logits must be on MPS");
    TORCH_CHECK(draft_logits.dtype() == torch::kFloat32, "draft_logits must be float32");
    TORCH_CHECK(draft_logits.is_contiguous(), "draft_logits must be contiguous");

    uint32_t K = draft_logits.size(0);
    uint32_t V = draft_logits.size(1);

    auto draft_tokens_int = draft_tokens.to(torch::kInt32).contiguous();

    auto p_draft = torch::empty_like(draft_logits);
    auto p_target = torch::empty_like(target_logits);
    auto first_reject = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));
    auto adjusted_probs = torch::zeros({(int64_t)V}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

    id<MTLComputePipelineState> softmax_pipe = get_softmax_pipeline();
    id<MTLComputePipelineState> verify_pipe = get_verify_pipeline();

    NSUInteger tg_size = 256;
    NSUInteger shared_mem = tg_size * sizeof(float);

    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            // --- Softmax pass ---
            {
                id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

                [enc setComputePipelineState:softmax_pipe];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(draft_logits)  offset:draft_logits.storage_offset() * draft_logits.element_size()  atIndex:0];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(target_logits) offset:target_logits.storage_offset() * target_logits.element_size() atIndex:1];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(p_draft)       offset:p_draft.storage_offset() * p_draft.element_size()       atIndex:2];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(p_target)      offset:p_target.storage_offset() * p_target.element_size()     atIndex:3];
                [enc setBytes:&V length:sizeof(uint32_t) atIndex:4];
                [enc setThreadgroupMemoryLength:shared_mem atIndex:0];

                MTLSize grid = MTLSizeMake(K, 1, 1);
                MTLSize tg = MTLSizeMake(tg_size, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }

            // End current encoder, commit, start new one for verify
            stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);

            // --- Verify pass ---
            {
                id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

                [enc setComputePipelineState:verify_pipe];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(p_draft)          offset:p_draft.storage_offset() * p_draft.element_size()          atIndex:0];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(p_target)         offset:p_target.storage_offset() * p_target.element_size()        atIndex:1];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(draft_tokens_int) offset:draft_tokens_int.storage_offset() * draft_tokens_int.element_size() atIndex:2];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(rand_uniform)     offset:rand_uniform.storage_offset() * rand_uniform.element_size() atIndex:3];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(first_reject)     offset:first_reject.storage_offset() * first_reject.element_size() atIndex:4];
                [enc setBuffer:at::native::mps::getMTLBufferStorage(adjusted_probs)   offset:adjusted_probs.storage_offset() * adjusted_probs.element_size() atIndex:5];
                [enc setBytes:&K length:sizeof(uint32_t) atIndex:6];
                [enc setBytes:&V length:sizeof(uint32_t) atIndex:7];

                MTLSize grid = MTLSizeMake(1, 1, 1);
                MTLSize tg = MTLSizeMake(1, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }
        }
    });

    return std::make_tuple(first_reject, adjusted_probs);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("speculative_verify", &speculative_verify_metal,
          "Fused speculative decoding verification (Metal)",
          py::arg("draft_logits"), py::arg("target_logits"),
          py::arg("draft_tokens"), py::arg("rand_uniform"));
}
