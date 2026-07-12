/**
 * @file image_utils.cpp
 * @brief 图像预处理函数实现
 *        Implementations of image pre-processing utility functions
 *
 * 包含 NV12→BGR（BT.601 标准）、双线性缩放、Letterbox 填充和 CHW Blob 生成。
 * Contains NV12→BGR (BT.601 standard), bilinear scaling, Letterbox padding,
 * and CHW blob generation.
 */
#include "image_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace image_utils {

/**
 * @brief 将浮点值钳位到 [0, 255] 并转为 uint8_t
 *        Clamp a float value to [0, 255] and convert to uint8_t
 */
inline uint8_t Clamp(float val) {
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return static_cast<uint8_t>(val);
}

bool NV12ToBGR(const uint8_t* nv12_data, int width, int height,
               int y_stride, int uv_stride, uint8_t* bgr_data) {
    if (!nv12_data || !bgr_data || width <= 0 || height <= 0 ||
        (width & 1) != 0 || (height & 1) != 0 ||
        y_stride < width || uv_stride < width) {
        return false;
    }
    const uint8_t* y_plane = nv12_data;
    const uint8_t* uv_plane = nv12_data + y_stride * height;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int y_idx = y * y_stride + x;
            // NV12 中两个 Y 共享一对 UV（2x2 块），UV 按交错排列
            // In NV12, every 2x2 Y block shares one U/V pair, stored interleaved
            int uv_idx = (y / 2) * uv_stride + (x / 2) * 2;
            
            uint8_t Y = y_plane[y_idx];
            uint8_t U = uv_plane[uv_idx];
            uint8_t V = uv_plane[uv_idx + 1];
            
            /* 标准 BT.601 YUV→RGB/BGR 转换系数
             * Standard BT.601 YUV to RGB/BGR conversion coefficients
             *   R = Y               + 1.402   × (V - 128)
             *   G = Y - 0.344136 × (U - 128) - 0.714136 × (V - 128)
             *   B = Y + 1.772   × (U - 128)
             */
            float r = Y + 1.402f * (V - 128);
            float g = Y - 0.344136f * (U - 128) - 0.714136f * (V - 128);
            float b = Y + 1.772f * (U - 128);
            
            // 输出为 BGR 顺序（OpenCV 风格） / Output in BGR order (OpenCV-style)
            int bgr_idx = (y * width + x) * 3;
            bgr_data[bgr_idx] = Clamp(b);
            bgr_data[bgr_idx + 1] = Clamp(g);
            bgr_data[bgr_idx + 2] = Clamp(r);
        }
    }
    return true;
}

void BilinearResize(const uint8_t* src, int src_w, int src_h, int src_stride,
                    uint8_t* dst, int dst_w, int dst_h, int dst_stride) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return;
    // 计算每个目标像素对应的源图像缩放比例
    // Compute the scale factor from source to target for each pixel
    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;
    
    for (int dy = 0; dy < dst_h; ++dy) {
        /* 将目标像素中心映射回源图像坐标（对齐像素中心）
         * Map target pixel center back to source coordinates (pixel-center alignment)
         *   sy = (dy + 0.5) * scale_y - 0.5
         */
        float sy = (dy + 0.5f) * scale_y - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = y0 + 1;
        float dy_val = sy - y0;  // 垂直方向插值权重 / vertical interpolation weight
        
        // Clamp 到有效范围 / Clamp to valid range
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));
        
        for (int dx = 0; dx < dst_w; ++dx) {
            float sx = (dx + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = x0 + 1;
            float dx_val = sx - x0;  // 水平方向插值权重 / horizontal interpolation weight
            
            x0 = std::max(0, std::min(x0, src_w - 1));
            x1 = std::max(0, std::min(x1, src_w - 1));
            
            // 对每个 BGR 通道分别做双线性插值
            // Bilinear interpolation on each BGR channel independently
            for (int c = 0; c < 3; ++c) {
                // 四个邻近像素的值 / Four neighbor pixel values
                float v00 = src[y0 * src_stride + x0 * 3 + c];
                float v01 = src[y0 * src_stride + x1 * 3 + c];
                float v10 = src[y1 * src_stride + x0 * 3 + c];
                float v11 = src[y1 * src_stride + x1 * 3 + c];
                
                /* 双线性插值公式 / Bilinear interpolation formula
                 * val = (1-dx)(1-dy)*v00 + dx(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11
                 */
                float val = (1.0f - dx_val) * (1.0f - dy_val) * v00 +
                            dx_val * (1.0f - dy_val) * v01 +
                            (1.0f - dx_val) * dy_val * v10 +
                            dx_val * dy_val * v11;
                
                dst[dy * dst_stride + dx * 3 + c] = static_cast<uint8_t>(std::max(0.0f, std::min(val, 255.0f)));
            }
        }
    }
}

void Letterbox(const face_rec::Image& src, int target_size, uint8_t* dst,
               LetterboxInfo* info, std::vector<uint8_t>* resize_buffer) {
    int w = src.width;
    int h = src.height;
    // 按宽高比计算缩放比例，取较小值保证图像完全包含在目标区域内
    // Compute scale factor preserving aspect ratio, using min to fit inside target
    float scale = std::min(static_cast<float>(target_size) / w, static_cast<float>(target_size) / h);
    int nw = static_cast<int>(w * scale);
    int nh = static_cast<int>(h * scale);
    
    // 计算边缘填充偏移，使缩放后的图像居中
    // Compute padding offsets to center the scaled image
    int pad_x = (target_size - nw) / 2;
    int pad_y = (target_size - nh) / 2;
    
    // 用填充色（灰色 114）初始化整个目标缓冲区
    // Fill destination buffer with padding color (gray 114)
    std::memset(dst, 114, target_size * target_size * 3);
    
    // 创建临时缓冲区保存缩放后的图像区域
    // Create temporary buffer for the resized image region
    std::vector<uint8_t> local_resized;
    std::vector<uint8_t>& resized = resize_buffer ? *resize_buffer : local_resized;
    resized.resize(static_cast<size_t>(nw) * nh * 3);
    BilinearResize(src.data, w, h, src.stride, resized.data(), nw, nh, nw * 3);
    
    // 将缩放后的图像复制到目标缓冲区的居中位置
    // Copy the resized image to the centered position in the destination buffer
    for (int y = 0; y < nh; ++y) {
        uint8_t* dst_ptr = dst + ((y + pad_y) * target_size + pad_x) * 3;
        const uint8_t* src_ptr = resized.data() + (y * nw * 3);
        std::memcpy(dst_ptr, src_ptr, nw * 3);
    }
    
    if (info) {
        info->scale = scale;
        info->pad_x = static_cast<float>(pad_x);
        info->pad_y = static_cast<float>(pad_y);
    }
}

bool BlobFromImage(const uint8_t* img_data, int width, int height, float* blob, size_t blob_size,
                   bool swap_rgb, float mean, float std) {
    if (!img_data || !blob || width <= 0 || height <= 0) return false;
    if (blob_size < static_cast<size_t>(width * height * 3)) return false;
    
    // CHW 布局：三个通道各占一个连续区域
    // CHW layout: each channel occupies a contiguous region
    int area = width * height;
    float* channel_0 = blob;           // 第 0 通道（B 或 R）/ channel 0 (B or R)
    float* channel_1 = blob + area;    // 第 1 通道（G）/ channel 1 (G)
    float* channel_2 = blob + 2 * area; // 第 2 通道（R 或 B）/ channel 2 (R or B)
    
    for (int i = 0; i < area; ++i) {
        uint8_t b = img_data[i * 3 + 0];
        uint8_t g = img_data[i * 3 + 1];
        uint8_t r = img_data[i * 3 + 2];
        
        if (swap_rgb) {
            // BGR→RGB 交换：将 R 放入 ch0，B 放入 ch2
            // BGR→RGB swap: place R in ch0, B in ch2
            channel_0[i] = (static_cast<float>(r) - mean) / std;
            channel_1[i] = (static_cast<float>(g) - mean) / std;
            channel_2[i] = (static_cast<float>(b) - mean) / std;
        } else {
            // 保持 BGR 顺序：B→ch0, G→ch1, R→ch2
            // Keep BGR order: B→ch0, G→ch1, R→ch2
            channel_0[i] = (static_cast<float>(b) - mean) / std;
            channel_1[i] = (static_cast<float>(g) - mean) / std;
            channel_2[i] = (static_cast<float>(r) - mean) / std;
        }
    }
    return true;
}

}  // namespace image_utils
