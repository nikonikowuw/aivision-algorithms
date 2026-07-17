/**
 * @file input_buffer_validator.cpp
 * @brief Overflow-safe validation for CPU image descriptors.
 */

#include "input_buffer_validator.h"

#include <limits>

namespace safety_helmet {
namespace {

bool ComputeRequiredSize(uint32_t rows, uint32_t stride,
                         uint32_t final_row_bytes, size_t& required) {
    const size_t rows_before_last = static_cast<size_t>(rows - 1u);
    const size_t stride_size = static_cast<size_t>(stride);
    const size_t final_size = static_cast<size_t>(final_row_bytes);
    if (rows_before_last >
        (std::numeric_limits<size_t>::max() - final_size) / stride_size) {
        return false;
    }
    required = rows_before_last * stride_size + final_size;
    return true;
}

}  // namespace

bool ValidateCpuBufferDescriptor(const hw_buffer_desc_t& input,
                                 CpuBufferLayout& layout,
                                 std::string& error) {
    layout = CpuBufferLayout{};
    error.clear();

    if (input.buffer_type != HW_BUFFER_TYPE_DEFAULT ||
        input.buffer_owner != HW_BUFFER_OWNER_ENGINE) {
        error = "CPU input must be Engine-owned DEFAULT memory";
        return false;
    }
    if (!input.data) {
        error = "CPU input data pointer is null";
        return false;
    }
    if (input.width == 0 || input.height == 0 ||
        input.width > kMaxFrameDimension || input.height > kMaxFrameDimension) {
        error = "CPU input dimensions are outside the supported range";
        return false;
    }
    if (input.stride == 0 ||
        input.stride > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        error = "CPU input stride is outside the OpenCV range";
        return false;
    }

    uint32_t row_bytes = 0;
    uint32_t rows = 0;
    if (input.pixel_format == pixel_format::BGR24) {
        if (input.width > std::numeric_limits<uint32_t>::max() / 3u) {
            error = "BGR24 row size calculation overflow";
            return false;
        }
        row_bytes = input.width * 3u;
        rows = input.height;
    } else if (input.pixel_format == pixel_format::NV12) {
        if ((input.width & 1u) != 0u || (input.height & 1u) != 0u) {
            error = "NV12 dimensions must be even";
            return false;
        }
        row_bytes = input.width;
        rows = input.height + input.height / 2u;
    } else {
        error = "CPU input pixel format is not BGR24 or NV12";
        return false;
    }

    if (input.stride < row_bytes) {
        error = "CPU input stride is smaller than the logical row";
        return false;
    }

    size_t required = 0;
    if (!ComputeRequiredSize(rows, input.stride, row_bytes, required)) {
        error = "CPU input required size calculation overflow";
        return false;
    }
    if (input.size < required) {
        error = "CPU input buffer is smaller than the described frame";
        return false;
    }

    layout.required_size = required;
    layout.logical_row_bytes = row_bytes;
    layout.logical_rows = rows;
    return true;
}

}  // namespace safety_helmet
