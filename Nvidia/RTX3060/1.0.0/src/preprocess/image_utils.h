/**
 * @file image_utils.h
 * @brief GPU Face Recognition — Image preprocessing utilities.
 *        Resize, normalize, letterbox, NV12-to-BGR conversion.
 * @module Preprocessing Layer
 */

#ifndef GPU_FACE_RECOGNITION_IMAGE_UTILS_H
#define GPU_FACE_RECOGNITION_IMAGE_UTILS_H

#include "common/types.h"
#include <vector>
#include <cstdint>

namespace face_rec {

/**
 * @brief Convert NV12 (YUV420sp) image to BGR24 using BT.601 coefficients.
 */
void Nv12ToBgr(const uint8_t* nv12_data, int width, int height, int nv12_stride,
               uint8_t* bgr_data);

/**
 * @brief Resize BGR image using bilinear interpolation.
 */
void BilinearResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                    uint8_t* dst, int dst_w, int dst_h, int dst_channels = 3);

/**
 * @brief Letterbox resize: preserve aspect ratio, pad with gray (114).
 *        Returns the scale factor and padding offsets used.
 */
struct LetterboxResult {
    float scale;
    int pad_x;
    int pad_y;
};
LetterboxResult LetterboxResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                                uint8_t* dst, int dst_w, int dst_h);

/**
 * @brief Single-pass planar blob: BGR→RGB swap + float normalization + CHW layout.
 *        Normalization: (pixel - 127.5) / 127.5
 * @param bgr  Input BGR pixel data (HWC layout)
 * @param rgb_planar Output CHW float planar data (caller must pre-allocate)
 * @param width Image width
 * @param height Image height
 * @param mean RGB mean values (3 floats) or nullptr for default 127.5
 * @param std_val RGB std values (3 floats) or nullptr for default 127.5
 */
void BgrToPlanarFloat(const uint8_t* bgr, float* rgb_planar,
                      int width, int height,
                      const float* mean = nullptr, const float* std_val = nullptr);

/**
 * @brief Crop a region from a BGR image (non-owning copy).
 */
void CropImage(const uint8_t* src, int src_w, int src_h, int src_stride,
               uint8_t* dst, const CropParams& crop);

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_IMAGE_UTILS_H
