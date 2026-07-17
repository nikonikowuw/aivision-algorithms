// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Borrowed Apple native buffer validation.

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include "apple_buffer_view.h"

#include <limits>

namespace smoking {

ErrorCode ValidateAppleBuffer(const hw_buffer_desc_t& descriptor,
                              AppleBufferView& view, std::string& error) {
    view = AppleBufferView{};
    error.clear();
    @try {
        if (descriptor.buffer_type != HW_BUFFER_TYPE_APPLE_NATIVE ||
            descriptor.buffer_owner != HW_BUFFER_OWNER_ENGINE) {
            error = "Apple native input must be Engine-owned APPLE_NATIVE";
            return ErrorCode::kInvalidBuffer;
        }
        const hw_buffer_apple_t& apple = descriptor.plat.apple;
        if (apple.abi_version != HW_BUFFER_APPLE_ABI_VERSION ||
            apple.struct_size != sizeof(hw_buffer_apple_t)) {
            error = "Apple native ABI version or struct_size mismatch";
            return ErrorCode::kInvalidBuffer;
        }
        if (apple.buffer_kind == HW_BUFFER_APPLE_IOSURFACE) {
            error = "direct IOSurface input is not supported";
            return ErrorCode::kUnsupportedCapability;
        }
        if (apple.buffer_kind != HW_BUFFER_APPLE_CVPIXELBUFFER ||
            apple.native_handle == 0) {
            error = "Apple native input must contain a CVPixelBuffer handle";
            return ErrorCode::kInvalidBuffer;
        }
        if (apple.synchronization != 0) {
            error = "Apple native synchronization tokens are not supported";
            return ErrorCode::kUnsupportedCapability;
        }

        CVPixelBufferRef pixel_buffer = reinterpret_cast<CVPixelBufferRef>(
            static_cast<uintptr_t>(apple.native_handle));
        const size_t actual_width = CVPixelBufferGetWidth(pixel_buffer);
        const size_t actual_height = CVPixelBufferGetHeight(pixel_buffer);
        if (actual_width == 0 || actual_height == 0 ||
            actual_width > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            actual_height > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            descriptor.width != actual_width || descriptor.height != actual_height) {
            error = "CVPixelBuffer dimensions do not match the ABI descriptor";
            return ErrorCode::kInvalidBuffer;
        }

        const OSType actual_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
        if (actual_format != kPixelFormatBGRA &&
            actual_format != kPixelFormatNV12VR &&
            actual_format != kPixelFormatNV12FR) {
            error = "CVPixelBuffer FourCC is unsupported";
            return ErrorCode::kUnsupportedCapability;
        }
        if (descriptor.pixel_format != actual_format ||
            apple.pixel_format != actual_format) {
            error = "CVPixelBuffer FourCC does not match descriptor metadata";
            return ErrorCode::kInvalidBuffer;
        }

        const bool bgra = actual_format == kPixelFormatBGRA;
        const bool planar = CVPixelBufferIsPlanar(pixel_buffer);
        const size_t actual_plane_count = CVPixelBufferGetPlaneCount(pixel_buffer);
        const uint32_t logical_plane_count = bgra ? 1u : 2u;
        if ((bgra && (planar || actual_plane_count != 0)) ||
            (!bgra && (!planar || actual_plane_count != 2)) ||
            apple.plane_count != logical_plane_count) {
            error = "CVPixelBuffer plane layout does not match the payload";
            return ErrorCode::kInvalidBuffer;
        }

        uint32_t strides[2] = {0, 0};
        if (bgra) {
            const size_t stride = CVPixelBufferGetBytesPerRow(pixel_buffer);
            if (stride > std::numeric_limits<uint32_t>::max() ||
                apple.plane_stride[0] != stride ||
                apple.plane_stride[1] != 0) {
                error = "BGRA CVPixelBuffer stride metadata mismatch";
                return ErrorCode::kInvalidBuffer;
            }
            strides[0] = static_cast<uint32_t>(stride);
        } else {
            for (size_t plane = 0; plane < 2; ++plane) {
                const size_t stride =
                    CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, plane);
                if (stride > std::numeric_limits<uint32_t>::max() ||
                    apple.plane_stride[plane] != stride) {
                    error = "NV12 CVPixelBuffer plane stride metadata mismatch";
                    return ErrorCode::kInvalidBuffer;
                }
                strides[plane] = static_cast<uint32_t>(stride);
            }
        }
        if (descriptor.stride != 0 && descriptor.stride != strides[0]) {
            error = "top-level stride does not match CVPixelBuffer plane 0";
            return ErrorCode::kInvalidBuffer;
        }

        view.native_handle = apple.native_handle;
        view.width = static_cast<int32_t>(actual_width);
        view.height = static_cast<int32_t>(actual_height);
        view.pixel_format = actual_format;
        view.plane_count = logical_plane_count;
        view.plane_stride[0] = strides[0];
        view.plane_stride[1] = strides[1];
        return ErrorCode::kSuccess;
    } @catch (NSException* exception) {
        error = std::string("CVPixelBuffer validation NSException: ") +
                [[exception description] UTF8String];
        return ErrorCode::kInvalidBuffer;
    }
}

}  // namespace smoking
