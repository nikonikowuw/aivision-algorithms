/**
 * @file input_buffer_validator.h
 * @brief Validation contract for Engine-owned CPU image descriptors.
 */

#ifndef SAFETY_HELMET_INPUT_BUFFER_VALIDATOR_H
#define SAFETY_HELMET_INPUT_BUFFER_VALIDATOR_H

#include "algo/abi_contract.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace safety_helmet {

namespace pixel_format {

/** Constructs the little-endian FourCC values used by the Engine ABI. */
constexpr uint32_t MakeFourCC(char a, char b, char c, char d) {
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8u) |
           (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16u) |
           (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24u);
}

constexpr uint32_t BGR24 = MakeFourCC('B', 'G', 'R', '3');
constexpr uint32_t NV12 = MakeFourCC('N', 'V', '1', '2');

}  // namespace pixel_format

/** Limits image dimensions before conversion to OpenCV signed dimensions. */
constexpr uint32_t kMaxFrameDimension = 16384u;

/** Describes the validated logical layout without taking ownership of memory. */
struct CpuBufferLayout {
    size_t required_size = 0;
    uint32_t logical_row_bytes = 0;
    uint32_t logical_rows = 0;
};

/**
 * Validates an Engine-owned BGR24 or NV12 CPU descriptor.
 *
 * The check rejects unsupported ownership/types, invalid dimensions and
 * formats, undersized strides/buffers, odd NV12 dimensions, and every size
 * calculation that cannot be represented by size_t.
 *
 * @param input Descriptor borrowed for the synchronous inference call.
 * @param layout Receives the validated logical layout on success.
 * @param error Receives a stable diagnostic suitable for internal logging.
 * @return true only when OpenCV can safely view the described memory.
 */
bool ValidateCpuBufferDescriptor(const hw_buffer_desc_t& input,
                                 CpuBufferLayout& layout,
                                 std::string& error);

}  // namespace safety_helmet

#endif  // SAFETY_HELMET_INPUT_BUFFER_VALIDATOR_H
