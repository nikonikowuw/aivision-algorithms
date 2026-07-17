// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Borrowed Apple native buffer validation.

#pragma once

#include <cstdint>
#include <string>

#include <algo/abi_contract.h>

#include "../common/types.h"

namespace smoking {

struct AppleBufferView {
    uint64_t native_handle = 0;
    int32_t width = 0;
    int32_t height = 0;
    uint32_t pixel_format = 0;
    uint32_t plane_count = 0;
    uint32_t plane_stride[2] = {0, 0};
};

/**
 * @brief Validate and borrow a CVPixelBuffer for the synchronous ABI call.
 * The returned view never retains, releases, or stores the native object.
 */
ErrorCode ValidateAppleBuffer(const hw_buffer_desc_t& descriptor,
                              AppleBufferView& view, std::string& error);

}  // namespace smoking
