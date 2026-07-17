// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Metal and BGR24 preprocessing implementation.

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include "metal_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace smoking {
namespace {

struct PreprocessParams {
    simd_float4 source_rect;
    simd_float4 content;
    uint32_t full_range;
};

bool BuildTransform(const RectF& requested_roi, int32_t frame_width,
                    int32_t frame_height, int32_t model_width,
                    int32_t model_height, ImageTransform& transform,
                    std::string& error) {
    RectF roi = requested_roi;
    roi.ClipToFrame(static_cast<float>(frame_width),
                    static_cast<float>(frame_height));
    if (!IsValidRect(roi) || roi.width < 1.0f || roi.height < 1.0f) {
        error = "preprocess ROI is empty or outside the frame";
        return false;
    }
    transform.source_region_in_frame = roi;
    transform.frame_width = frame_width;
    transform.frame_height = frame_height;
    transform.model_width = model_width;
    transform.model_height = model_height;
    transform.scale = std::min(static_cast<float>(model_width) / roi.width,
                               static_cast<float>(model_height) / roi.height);
    transform.pad_x = (model_width - roi.width * transform.scale) * 0.5f;
    transform.pad_y = (model_height - roi.height * transform.scale) * 0.5f;
    return std::isfinite(transform.scale) && transform.scale > 0.0f;
}

}  // namespace

struct MetalPreprocessor::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> bgra_pipeline = nil;
    id<MTLComputePipelineState> nv12_pipeline = nil;
    CVMetalTextureCacheRef texture_cache = nullptr;
    CVPixelBufferPoolRef output_pool = nullptr;
    CVPixelBufferRef current_output = nullptr;
    int32_t model_width = 0;
    int32_t model_height = 0;

    bool CreateOutput(std::string& error) {
        if (current_output) {
            CVPixelBufferRelease(current_output);
            current_output = nullptr;
        }
        const CVReturn status =
            CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault,
                                               output_pool, &current_output);
        if (status != kCVReturnSuccess || !current_output) {
            error = "failed to allocate prepared CVPixelBuffer from pool";
            return false;
        }
        return true;
    }
};

MetalPreprocessor::MetalPreprocessor() : impl_(std::make_unique<Impl>()) {}
MetalPreprocessor::~MetalPreprocessor() { Destroy(); }

bool MetalPreprocessor::Initialize(int32_t model_width, int32_t model_height,
                                   const std::string& metallib_path,
                                   std::string& error) {
    error.clear();
    Destroy();
    @autoreleasepool {
        @try {
            if (model_width <= 0 || model_height <= 0 || metallib_path.empty()) {
                error = "invalid preprocess dimensions or metallib path";
                return false;
            }
            NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
            BOOL is_directory = NO;
            if (!path || ![[NSFileManager defaultManager] fileExistsAtPath:path
                                                               isDirectory:&is_directory] ||
                is_directory) {
                error = "Metal shader library is missing: " + metallib_path;
                return false;
            }

            impl_->device = MTLCreateSystemDefaultDevice();
            impl_->queue = [impl_->device newCommandQueue];
            if (!impl_->device || !impl_->queue) {
                error = "Metal device or command queue is unavailable";
                return false;
            }

            NSError* ns_error = nil;
            id<MTLLibrary> library = [impl_->device
                newLibraryWithURL:[NSURL fileURLWithPath:path] error:&ns_error];
            if (!library || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "failed to load Metal library";
                return false;
            }
            id<MTLFunction> bgra = [library newFunctionWithName:@"preprocess_bgra"];
            id<MTLFunction> nv12 = [library newFunctionWithName:@"preprocess_nv12"];
            if (!bgra || !nv12) {
                error = "required preprocess_bgra/preprocess_nv12 kernels are missing";
                return false;
            }
            impl_->bgra_pipeline =
                [impl_->device newComputePipelineStateWithFunction:bgra error:&ns_error];
            if (!impl_->bgra_pipeline || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "failed to create BGRA pipeline";
                return false;
            }
            ns_error = nil;
            impl_->nv12_pipeline =
                [impl_->device newComputePipelineStateWithFunction:nv12 error:&ns_error];
            if (!impl_->nv12_pipeline || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "failed to create NV12 pipeline";
                return false;
            }

            if (CVMetalTextureCacheCreate(kCFAllocatorDefault, nullptr,
                                          impl_->device, nullptr,
                                          &impl_->texture_cache) != kCVReturnSuccess) {
                error = "failed to create CVMetalTextureCache";
                return false;
            }

            NSDictionary* attributes = @{
                (NSString*)kCVPixelBufferPixelFormatTypeKey :
                    @(kCVPixelFormatType_32BGRA),
                (NSString*)kCVPixelBufferWidthKey : @(model_width),
                (NSString*)kCVPixelBufferHeightKey : @(model_height),
                (NSString*)kCVPixelBufferMetalCompatibilityKey : @YES,
                (NSString*)kCVPixelBufferIOSurfacePropertiesKey : @{},
            };
            if (CVPixelBufferPoolCreate(kCFAllocatorDefault, nullptr,
                                        (__bridge CFDictionaryRef)attributes,
                                        &impl_->output_pool) != kCVReturnSuccess) {
                error = "failed to create prepared-image CVPixelBufferPool";
                return false;
            }
            impl_->model_width = model_width;
            impl_->model_height = model_height;
            return true;
        } @catch (NSException* exception) {
            error = std::string("Metal initialization NSException: ") +
                    [[exception description] UTF8String];
            Destroy();
            return false;
        }
    }
}

void MetalPreprocessor::Destroy() {
    if (!impl_) return;
    if (impl_->current_output) {
        CVPixelBufferRelease(impl_->current_output);
        impl_->current_output = nullptr;
    }
    if (impl_->output_pool) {
        CFRelease(impl_->output_pool);
        impl_->output_pool = nullptr;
    }
    if (impl_->texture_cache) {
        CFRelease(impl_->texture_cache);
        impl_->texture_cache = nullptr;
    }
    impl_->bgra_pipeline = nil;
    impl_->nv12_pipeline = nil;
    impl_->queue = nil;
    impl_->device = nil;
    impl_->model_width = 0;
    impl_->model_height = 0;
}

bool MetalPreprocessor::ProcessNative(
    uint64_t cv_pixel_buffer_handle, int32_t frame_width, int32_t frame_height,
    uint32_t pixel_format, const RectF& roi, PreparedImage& prepared,
    std::string& error) {
    prepared = PreparedImage{};
    error.clear();
    @autoreleasepool {
        @try {
            if (!impl_->texture_cache || !impl_->output_pool ||
                cv_pixel_buffer_handle == 0) {
                error = "Metal preprocessor is not initialized or input is null";
                return false;
            }
            if (pixel_format != kPixelFormatBGRA &&
                pixel_format != kPixelFormatNV12VR &&
                pixel_format != kPixelFormatNV12FR) {
                error = "unsupported native pixel format";
                return false;
            }
            if (!BuildTransform(roi, frame_width, frame_height,
                                impl_->model_width, impl_->model_height,
                                prepared.transform, error) ||
                !impl_->CreateOutput(error)) {
                return false;
            }

            CVPixelBufferRef source = reinterpret_cast<CVPixelBufferRef>(
                static_cast<uintptr_t>(cv_pixel_buffer_handle));
            CVMetalTextureRef source0 = nullptr;
            CVMetalTextureRef source1 = nullptr;
            CVMetalTextureRef output_texture = nullptr;
            const bool is_bgra = pixel_format == kPixelFormatBGRA;
            CVReturn status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, impl_->texture_cache, source, nullptr,
                is_bgra ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatR8Unorm,
                is_bgra ? static_cast<size_t>(frame_width)
                        : CVPixelBufferGetWidthOfPlane(source, 0),
                is_bgra ? static_cast<size_t>(frame_height)
                        : CVPixelBufferGetHeightOfPlane(source, 0),
                0, &source0);
            if (status != kCVReturnSuccess || !source0) {
                error = "failed to create Metal texture for source plane 0";
                return false;
            }
            if (!is_bgra) {
                status = CVMetalTextureCacheCreateTextureFromImage(
                    kCFAllocatorDefault, impl_->texture_cache, source, nullptr,
                    MTLPixelFormatRG8Unorm,
                    CVPixelBufferGetWidthOfPlane(source, 1),
                    CVPixelBufferGetHeightOfPlane(source, 1), 1, &source1);
                if (status != kCVReturnSuccess || !source1) {
                    CFRelease(source0);
                    error = "failed to create Metal texture for NV12 chroma plane";
                    return false;
                }
            }
            status = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault, impl_->texture_cache, impl_->current_output,
                nullptr, MTLPixelFormatBGRA8Unorm, impl_->model_width,
                impl_->model_height, 0, &output_texture);
            if (status != kCVReturnSuccess || !output_texture) {
                if (source1) CFRelease(source1);
                CFRelease(source0);
                error = "failed to create Metal output texture";
                return false;
            }

            id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:is_bgra ? impl_->bgra_pipeline
                                                      : impl_->nv12_pipeline];
            [encoder setTexture:CVMetalTextureGetTexture(source0) atIndex:0];
            if (source1) {
                [encoder setTexture:CVMetalTextureGetTexture(source1) atIndex:1];
            }
            [encoder setTexture:CVMetalTextureGetTexture(output_texture) atIndex:2];

            const auto& transform = prepared.transform;
            PreprocessParams params{};
            params.source_rect = {transform.source_region_in_frame.x,
                                  transform.source_region_in_frame.y,
                                  transform.source_region_in_frame.width,
                                  transform.source_region_in_frame.height};
            params.content = {transform.pad_x, transform.pad_y,
                              transform.scale, 0.0f};
            params.full_range = pixel_format == kPixelFormatNV12FR ? 1u : 0u;
            [encoder setBytes:&params length:sizeof(params) atIndex:0];
            [encoder dispatchThreads:MTLSizeMake(impl_->model_width,
                                                 impl_->model_height, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (source1) CFRelease(source1);
            CFRelease(source0);
            CFRelease(output_texture);
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                error = command_buffer.error
                            ? [[command_buffer.error localizedDescription] UTF8String]
                            : "Metal preprocess command did not complete";
                return false;
            }
            prepared.pixel_buffer = reinterpret_cast<uint64_t>(
                impl_->current_output);
            return true;
        } @catch (NSException* exception) {
            prepared = PreparedImage{};
            error = std::string("Metal preprocess NSException: ") +
                    [[exception description] UTF8String];
            return false;
        }
    }
}

bool MetalPreprocessor::ProcessBGR24(
    const uint8_t* frame, int32_t frame_width, int32_t frame_height,
    int32_t stride, const RectF& roi, PreparedImage& prepared,
    std::string& error) {
    prepared = PreparedImage{};
    error.clear();
    @autoreleasepool {
        @try {
            if (!frame || stride < frame_width * 3 || !impl_->output_pool) {
                error = "invalid BGR24 input or uninitialized output pool";
                return false;
            }
            if (!BuildTransform(roi, frame_width, frame_height,
                                impl_->model_width, impl_->model_height,
                                prepared.transform, error) ||
                !impl_->CreateOutput(error)) {
                return false;
            }
            if (CVPixelBufferLockBaseAddress(impl_->current_output, 0) !=
                kCVReturnSuccess) {
                error = "failed to lock prepared BGR24 output";
                return false;
            }

            auto* destination = static_cast<uint8_t*>(
                CVPixelBufferGetBaseAddress(impl_->current_output));
            const size_t destination_stride =
                CVPixelBufferGetBytesPerRow(impl_->current_output);
            const auto& transform = prepared.transform;
            for (int32_t y = 0; y < impl_->model_height; ++y) {
                auto* row = destination + static_cast<size_t>(y) * destination_stride;
                for (int32_t x = 0; x < impl_->model_width; ++x) {
                    uint8_t* pixel = row + static_cast<size_t>(x) * 4;
                    const float source_x = transform.source_region_in_frame.x +
                        (x + 0.5f - transform.pad_x) / transform.scale;
                    const float source_y = transform.source_region_in_frame.y +
                        (y + 0.5f - transform.pad_y) / transform.scale;
                    if (source_x < transform.source_region_in_frame.Left() ||
                        source_x >= transform.source_region_in_frame.Right() ||
                        source_y < transform.source_region_in_frame.Top() ||
                        source_y >= transform.source_region_in_frame.Bottom()) {
                        pixel[0] = pixel[1] = pixel[2] = 114;
                        pixel[3] = 255;
                        continue;
                    }

                    const int32_t x0 = std::clamp(
                        static_cast<int32_t>(std::floor(source_x)), 0,
                        frame_width - 1);
                    const int32_t y0 = std::clamp(
                        static_cast<int32_t>(std::floor(source_y)), 0,
                        frame_height - 1);
                    const int32_t x1 = std::min(x0 + 1, frame_width - 1);
                    const int32_t y1 = std::min(y0 + 1, frame_height - 1);
                    const float fx = source_x - x0;
                    const float fy = source_y - y0;
                    for (int channel = 0; channel < 3; ++channel) {
                        const float p00 = frame[static_cast<size_t>(y0) * stride + x0 * 3 + channel];
                        const float p10 = frame[static_cast<size_t>(y0) * stride + x1 * 3 + channel];
                        const float p01 = frame[static_cast<size_t>(y1) * stride + x0 * 3 + channel];
                        const float p11 = frame[static_cast<size_t>(y1) * stride + x1 * 3 + channel];
                        const float top = p00 + (p10 - p00) * fx;
                        const float bottom = p01 + (p11 - p01) * fx;
                        pixel[channel] = static_cast<uint8_t>(std::clamp(
                            top + (bottom - top) * fy, 0.0f, 255.0f));
                    }
                    pixel[3] = 255;
                }
            }
            CVPixelBufferUnlockBaseAddress(impl_->current_output, 0);
            prepared.pixel_buffer = reinterpret_cast<uint64_t>(
                impl_->current_output);
            return true;
        } @catch (NSException* exception) {
            prepared = PreparedImage{};
            error = std::string("BGR24 preprocess NSException: ") +
                    [[exception description] UTF8String];
            return false;
        }
    }
}

}  // namespace smoking
