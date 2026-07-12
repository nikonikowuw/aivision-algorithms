/**
 * @file image_utils.h
 * @brief 图像预处理工具函数声明
 *        Declarations of image pre-processing utility functions
 *
 * 提供 NV12→BGR 转换、双线性缩放、Letterbox 填充及 CHW Blob 生成等基础图像操作。
 * Provides NV12→BGR conversion, bilinear resize, Letterbox padding,
 * and CHW blob generation for model inference.
 */
#ifndef FACE_RECOGNITION_IMAGE_UTILS_H
#define FACE_RECOGNITION_IMAGE_UTILS_H

#include "common/types.h"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace image_utils {

/**
 * @brief Letterbox 变换信息 / Letterbox transform metadata
 *
 * Letterbox 处理时会保持宽高比缩放图像，并在边缘填充灰色像素。
 * 该结构体记录缩放比例和填充偏移，供后续坐标反算使用。
 * Records the scaling factor and padding offsets applied during letterbox,
 * used to map detection coordinates back to the original image.
 */
struct LetterboxInfo {
    float scale = 1.0f;   // 缩放比例 / scaling factor
    float pad_x = 0.0f;   // 水平填充像素数 / horizontal padding (pixels)
    float pad_y = 0.0f;   // 垂直填充像素数 / vertical padding (pixels)
};

/**
 * @brief 将 NV12 原始内存布局转换为打包 BGR (HWC)
 *        Convert NV12 raw memory layout to packed BGR (HWC) layout
 * @param nv12_data  NV12 输入数据指针 / pointer to NV12 input data
 * @param width      图像宽度 / image width
 * @param height     图像高度 / image height
 * @param bgr_data   BGR 输出缓冲区 (width*height*3 bytes) / output BGR buffer
 * @return true 成功 / success, false 参数无效 / invalid parameters
 */
bool NV12ToBGR(const uint8_t* nv12_data, int width, int height, uint8_t* bgr_data);

/**
 * @brief 对源 BGR 图像做双线性插值缩放
 *        Bilinear resize of source BGR image to target BGR image
 * @param src       源图数据 / source image data
 * @param src_w     源图宽度 / source width
 * @param src_h     源图高度 / source height
 * @param src_stride 源图行跨度（字节） / source row stride (bytes)
 * @param dst       目标图数据 / destination image data
 * @param dst_w     目标宽度 / destination width
 * @param dst_h     目标高度 / destination height
 * @param dst_stride 目标行跨度（字节） / destination row stride (bytes)
 */
void BilinearResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                    uint8_t* dst, int dst_w, int dst_h, int dst_stride);

/**
 * @brief Letterbox 缩放：保持宽高比，用 114（灰色）填充至 target_size×target_size
 *        Letterbox resize: maintains aspect ratio, pads to target_size with gray (114)
 * @param src         源图像 / source image
 * @param target_size 目标尺寸（宽=高） / target size (square)
 * @param dst         输出缓冲区 (target_size*target_size*3 bytes) / output buffer
 * @param info        [out] Letterbox 变换信息 / letterbox transform info (optional)
 */
void Letterbox(const face_rec::Image& src, int target_size, uint8_t* dst, LetterboxInfo* info);

/**
 * @brief 将 BGR HWC 图像转换为 float CHW Blob，支持通道交换与归一化
 *        Formats BGR HWC image to float CHW blob for model input
 * @param img_data  输入 BGR 图像数据 / input BGR image data
 * @param width     图像宽度 / image width
 * @param height    图像高度 / image height
 * @param blob      [out] float 类型输出 Blob / output float blob
 * @param blob_size 输出 Blob 大小（元素数） / output blob size (number of floats)
 * @param swap_rgb  是否交换 B↔R 通道（BGR→RGB 或保持 BGR） / swap B↔R channels
 * @param mean      归一化均值 / normalization mean
 * @param std       归一化标准差 / normalization standard deviation
 * @return true 成功 / success, false 参数无效 / invalid parameters
 */
bool BlobFromImage(const uint8_t* img_data, int width, int height, float* blob, size_t blob_size,
                   bool swap_rgb = false, float mean = 0.0f, float std = 1.0f);

}  // namespace image_utils

#endif // FACE_RECOGNITION_IMAGE_UTILS_H
