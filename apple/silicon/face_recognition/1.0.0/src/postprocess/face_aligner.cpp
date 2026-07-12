/**
 * @file face_aligner.cpp
 * @brief 人脸对齐实现（相似变换 + 双线性插值）
 *        Face alignment implementation (similarity transform + bilinear interpolation)
 *
 * 使用最小二乘法求解从源关键点到参考关键点的最佳相似变换，
 * 然后通过逆映射双线性插值生成 112×112 的对齐人脸图像。
 * Solves an optimal similarity transform from source landmarks to reference
 * landmarks via least squares, then generates a 112×112 aligned face image
 * through inverse-mapping bilinear interpolation.
 */
#include "face_aligner.h"
#include "preprocess/image_utils.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace face_rec {

bool InsightFaceAligner::Align(const Image& src, const std::array<Point, 5>& landmarks, uint8_t* dst_data) {
    if (!src.data || !dst_data || src.width <= 0 || src.height <= 0) return false;
    
    // ---- 第 1 步：求解相似变换参数 (a, b, tx, ty) ----
    // Step 1: Solve for similarity transform parameters (a, b, tx, ty)
    //
    // 相似变换模型：相似变换可以表示为：
    // Similarity transform model:
    //   | x_dst |   | a  -b | | x_src |   | tx |
    //   | y_dst | = | b   a | | y_src | + | ty |
    //
    // 其中 (a, b) 包含旋转和缩放信息，(tx, ty) 为平移量。
    // Where (a, b) encode rotation and scale, (tx, ty) is translation.
    
    // 计算源点和目标点的质心 / Compute centroids of source and target points
    float mean_src_x = 0.0f, mean_src_y = 0.0f;
    float mean_dst_x = 0.0f, mean_dst_y = 0.0f;
    
    for (int i = 0; i < 5; ++i) {
        mean_src_x += landmarks[i].x;
        mean_src_y += landmarks[i].y;
        mean_dst_x += kRefPoints[i][0];
        mean_dst_y += kRefPoints[i][1];
    }
    
    mean_src_x /= 5.0f;
    mean_src_y /= 5.0f;
    mean_dst_x /= 5.0f;
    mean_dst_y /= 5.0f;
    
    // 使用去中心化坐标求解最小二乘最优 (a, b)
    // Solve least-squares optimal (a, b) using de-meaned coordinates
    float num_a = 0.0f;  // Σ(dx_src * dx_dst + dy_src * dy_dst)
    float num_b = 0.0f;  // Σ(dx_src * dy_dst - dy_src * dx_dst)
    float den = 0.0f;    // Σ(dx_src² + dy_src²)
    
    for (int i = 0; i < 5; ++i) {
        float dx_src = landmarks[i].x - mean_src_x;
        float dy_src = landmarks[i].y - mean_src_y;
        float dx_dst = kRefPoints[i][0] - mean_dst_x;
        float dy_dst = kRefPoints[i][1] - mean_dst_y;
        
        num_a += dx_src * dx_dst + dy_src * dy_dst;
        num_b += dx_src * dy_dst - dy_src * dx_dst;
        den += dx_src * dx_src + dy_src * dy_src;
    }
    
    float a = 1.0f;
    float b = 0.0f;
    float tx = 0.0f;
    float ty = 0.0f;
    bool solved = false;
    
    // 若分母过小（关键点退化），跳过相似变换求解，直接做简单缩放
    // If denominator is too small (degenerate landmarks), skip similarity transform
    if (den > 1e-6f) {
        a = num_a / den;
        b = num_b / den;
        // 恢复平移量 / Recover translation
        tx = mean_dst_x - (a * mean_src_x - b * mean_src_y);
        ty = mean_dst_y - (b * mean_src_x + a * mean_src_y);
        solved = true;
    }
    
    float det = a * a + b * b;
    // 如果求解失败或变换矩阵奇异，回退到简单双线性缩放
    // If solving failed or transform matrix is singular, fallback to plain resize
    if (!solved || det < 1e-6f) {
        image_utils::BilinearResize(src.data, src.width, src.height, src.stride, dst_data, 112, 112, 112 * 3);
        return true;
    }
    
    // ---- 第 2 步：使用逆映射双线性插值生成对齐人脸 ----
    // Step 2: Generate aligned face via inverse-mapping bilinear interpolation
    //
    // 对目标图像中的每个像素 (u, v)，通过逆变换找到其在源图中的对应位置 (x, y)，
    // 然后使用双线性插值采样。逆映射公式：
    // For each destination pixel (u, v), find its source location (x, y) via inverse
    // transform, then sample with bilinear interpolation:
    //   | x |   1   |  a  b | | u - tx |
    //   | y | = ----- | -b  a | | v - ty |
    //            det
    for (int v = 0; v < 112; ++v) {
        for (int u = 0; u < 112; ++u) {
            float du_t = u - tx;
            float dv_t = v - ty;
            float x = (a * du_t + b * dv_t) / det;
            float y = (-b * du_t + a * dv_t) / det;
            
            int dst_pixel_idx = (v * 112 + u) * 3;
            
            // 确保源图坐标在有效范围内（保留一个像素的边界用于插值）
            // Ensure source coordinates are within valid range (keep 1-pixel border for interpolation)
            if (x >= 0.0f && x < src.width - 1 && y >= 0.0f && y < src.height - 1) {
                int x0 = static_cast<int>(std::floor(x));
                int y0 = static_cast<int>(std::floor(y));
                float dx = x - x0;
                float dy = y - y0;
                
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                // 对每个颜色通道做双线性插值
                // Bilinear interpolation per color channel
                for (int c = 0; c < 3; ++c) {
                    float v00 = src.data[y0 * src.stride + x0 * 3 + c];
                    float v01 = src.data[y0 * src.stride + x1 * 3 + c];
                    float v10 = src.data[y1 * src.stride + x0 * 3 + c];
                    float v11 = src.data[y1 * src.stride + x1 * 3 + c];
                    
                    float val = (1.0f - dx) * (1.0f - dy) * v00 +
                                dx * (1.0f - dy) * v01 +
                                (1.0f - dx) * dy * v10 +
                                dx * dy * v11;
                    
                    dst_data[dst_pixel_idx + c] = static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
                }
            } else {
                // 超出源图范围的位置填充黑色背景
                // Out-of-bounds positions filled with black background
                dst_data[dst_pixel_idx] = 0;
                dst_data[dst_pixel_idx + 1] = 0;
                dst_data[dst_pixel_idx + 2] = 0;
            }
        }
    }
    
    return true;
}

} // namespace face_rec
