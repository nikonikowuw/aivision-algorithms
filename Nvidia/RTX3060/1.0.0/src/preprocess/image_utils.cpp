/**
 * @file image_utils.cpp
 * @brief GPU Face Recognition — Image preprocessing implementation.
 */

#include "image_utils.h"
#include <algorithm>
#include <cstring>

namespace face_rec {

void Nv12ToBgr(const uint8_t* nv12_data, int width, int height, int nv12_stride,
               uint8_t* bgr_data) {
    const uint8_t* y_plane = nv12_data;
    const uint8_t* uv_plane = nv12_data + nv12_stride * height;

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int y_val = y_plane[row * nv12_stride + col];
            int u_val = uv_plane[(row / 2) * nv12_stride + (col / 2) * 2];
            int v_val = uv_plane[(row / 2) * nv12_stride + (col / 2) * 2 + 1];

            // BT.601 conversion
            int c = y_val - 16;
            int d = u_val - 128;
            int e = v_val - 128;

            int b = (298 * c + 100 * d + 516 * e) >> 8;
            int g = (298 * c - 121 * d - 208 * e) >> 8;
            int r = (298 * c + 409 * d - 100 * e) >> 8;

            bgr_data[(row * width + col) * 3 + 0] = static_cast<uint8_t>(std::clamp(b, 0, 255));
            bgr_data[(row * width + col) * 3 + 1] = static_cast<uint8_t>(std::clamp(g, 0, 255));
            bgr_data[(row * width + col) * 3 + 2] = static_cast<uint8_t>(std::clamp(r, 0, 255));
        }
    }
}

void BilinearResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                    uint8_t* dst, int dst_w, int dst_h, int dst_channels) {
    float x_ratio = static_cast<float>(src_w) / dst_w;
    float y_ratio = static_cast<float>(src_h) / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            int src_x = static_cast<int>(x * x_ratio);
            int src_y = static_cast<int>(y * y_ratio);
            src_x = std::min(src_x, src_w - 1);
            src_y = std::min(src_y, src_h - 1);

            for (int c = 0; c < dst_channels; ++c) {
                dst[(y * dst_w + x) * dst_channels + c] =
                    src[(src_y * src_stride + src_x * 3) + c];
            }
        }
    }
}

LetterboxResult LetterboxResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                                uint8_t* dst, int dst_w, int dst_h) {
    // Fill with gray (114) — standard YOLO letterbox padding
    std::memset(dst, 114, dst_w * dst_h * 3);

    float scale = std::min(static_cast<float>(dst_w) / src_w,
                           static_cast<float>(dst_h) / src_h);
    int new_w = static_cast<int>(src_w * scale);
    int new_h = static_cast<int>(src_h * scale);
    int pad_x = (dst_w - new_w) / 2;
    int pad_y = (dst_h - new_h) / 2;

    // Bilinear resize into the center of the padded canvas
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int src_x = std::min(static_cast<int>(x / scale), src_w - 1);
            int src_y = std::min(static_cast<int>(y / scale), src_h - 1);

            int dst_idx = ((y + pad_y) * dst_w + (x + pad_x)) * 3;
            int src_idx = (src_y * src_stride + src_x * 3);

            dst[dst_idx + 0] = src[src_idx + 0];
            dst[dst_idx + 1] = src[src_idx + 1];
            dst[dst_idx + 2] = src[src_idx + 2];
        }
    }

    return {scale, pad_x, pad_y};
}

void BgrToPlanarFloat(const uint8_t* bgr, float* rgb_planar,
                      int width, int height,
                      const float* mean, const float* std_val) {
    float default_mean[3] = {127.5f, 127.5f, 127.5f};
    float default_std[3]  = {127.5f, 127.5f, 127.5f};
    const float* m = mean ? mean : default_mean;
    const float* s = std_val ? std_val : default_std;

    int plane_size = width * height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int src_idx = (y * width + x) * 3;
            // BGR → RGB order, normalize: (pixel - mean) / std
            rgb_planar[0 * plane_size + y * width + x] =
                (bgr[src_idx + 2] - m[0]) / s[0];  // R
            rgb_planar[1 * plane_size + y * width + x] =
                (bgr[src_idx + 1] - m[1]) / s[1];  // G
            rgb_planar[2 * plane_size + y * width + x] =
                (bgr[src_idx + 0] - m[2]) / s[2];  // B
        }
    }
}

void CropImage(const uint8_t* src, int src_w, int src_h, int src_stride,
               uint8_t* dst, const CropParams& crop) {
    int cx = std::max(0, std::min(crop.x, src_w));
    int cy = std::max(0, std::min(crop.y, src_h));
    int cw = std::min(crop.w, src_w - cx);
    int ch = std::min(crop.h, src_h - cy);

    for (int y = 0; y < ch; ++y) {
        std::memcpy(dst + y * cw * 3,
                    src + (cy + y) * src_stride * 3 + cx * 3,
                    cw * 3);
    }
}

} // namespace face_rec
