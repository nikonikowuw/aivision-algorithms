/**
 * @file test_input_buffer_validator.cpp
 * @brief Focused tests for CPU descriptor validation without loading a model.
 */

#include "common/input_buffer_validator.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

hw_buffer_desc_t MakeDescriptor(std::vector<uint8_t>& storage,
                                uint32_t width, uint32_t height,
                                uint32_t format, uint32_t stride) {
    hw_buffer_desc_t input{};
    input.data = storage.data();
    input.size = storage.size();
    input.width = width;
    input.height = height;
    input.pixel_format = format;
    input.stride = stride;
    input.buffer_type = HW_BUFFER_TYPE_DEFAULT;
    input.buffer_owner = HW_BUFFER_OWNER_ENGINE;
    return input;
}

}  // namespace

int main() {
    using safety_helmet::CpuBufferLayout;
    using safety_helmet::ValidateCpuBufferDescriptor;

    bool ok = true;
    CpuBufferLayout layout;
    std::string error;

    std::vector<uint8_t> bgr(32u * 4u);
    auto input = MakeDescriptor(bgr, 8u, 4u,
                                safety_helmet::pixel_format::BGR24, 32u);
    ok &= Expect(ValidateCpuBufferDescriptor(input, layout, error),
                 "valid padded BGR24 descriptor was rejected");
    ok &= Expect(layout.required_size == 120u,
                 "BGR24 required size ignored the final logical row width");

    input.stride = 23u;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "undersized BGR24 stride was accepted");

    input.stride = 32u;
    input.size = 119u;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "undersized BGR24 buffer was accepted");

    input.size = bgr.size();
    input.buffer_owner = HW_BUFFER_OWNER_ALGORITHM;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "Algorithm-owned CPU buffer was accepted");

    std::vector<uint8_t> nv12(16u * 9u);
    input = MakeDescriptor(nv12, 12u, 6u,
                           safety_helmet::pixel_format::NV12, 16u);
    ok &= Expect(ValidateCpuBufferDescriptor(input, layout, error),
                 "valid padded NV12 descriptor was rejected");
    ok &= Expect(layout.logical_rows == 9u && layout.required_size == 140u,
                 "NV12 required size or row count is incorrect");

    input.width = 11u;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "odd-width NV12 descriptor was accepted");

    input.width = 12u;
    input.height = 5u;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "odd-height NV12 descriptor was accepted");

    input.height = 6u;
    input.width = safety_helmet::kMaxFrameDimension + 1u;
    ok &= Expect(!ValidateCpuBufferDescriptor(input, layout, error),
                 "oversized dimensions were accepted");

    if (!ok) return 1;
    std::cout << "PASS: CPU buffer descriptor validation\n";
    return 0;
}
