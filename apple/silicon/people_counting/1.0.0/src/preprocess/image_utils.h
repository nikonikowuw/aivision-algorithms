/**
 * @file image_utils.h
 * @brief Image pre-processing utilities
 */

#ifndef PEOPLE_COUNTING_IMAGE_UTILS_H
#define PEOPLE_COUNTING_IMAGE_UTILS_H

#include "common/types.h"
#include <vector>
#include <cstdint>

namespace image_utils {

struct LetterboxInfo {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

bool NV12ToBGR(const uint8_t* nv12_data, int width, int height,
               int y_stride, int uv_stride, uint8_t* bgr_data);

void BilinearResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                    uint8_t* dst, int dst_w, int dst_h, int dst_stride);

void Letterbox(const people_count::Image& src, int target_size, uint8_t* dst,
               LetterboxInfo* info, std::vector<uint8_t>* resize_buffer = nullptr);

bool BlobFromImage(const uint8_t* img_data, int width, int height, float* blob, size_t blob_size,
                   bool swap_rgb = false, float mean = 0.0f, float std = 1.0f);

} // namespace image_utils

#endif // PEOPLE_COUNTING_IMAGE_UTILS_H
