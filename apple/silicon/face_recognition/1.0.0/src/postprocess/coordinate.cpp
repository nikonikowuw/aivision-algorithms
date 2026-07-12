/**
 * @file coordinate.cpp
 * @brief 坐标映射与 ROI 计算实现
 *        Implementations of coordinate mapping and ROI computation
 *
 * 实现两阶段坐标反算逻辑：先将 Letterbox 检测框变换回 local 坐标，
 * 再根据是否使用 ROI 裁剪叠加偏移映射回原始全图坐标。
 * Implements two-stage inverse coordinate mapping: first from letterbox to
 * local ROI, then (optionally) from ROI to the original full frame.
 */
#include "coordinate.h"
#include <algorithm>

namespace face_rec {

void CoordinateMapper::RestoreToOriginal(
    const image_utils::LetterboxInfo& letterbox,
    int orig_width, int orig_height,
    std::vector<DetectedObject>* objects) {
    if (!objects) return;
    
    for (auto& obj : *objects) {
        // 单阶段反算：去除 Letterbox 填充并除以缩放系数
        // 1-stage inverse: remove letterbox padding and divide by scale factor
        float x1 = (obj.bbox.x - letterbox.pad_x) / letterbox.scale;
        float y1 = (obj.bbox.y - letterbox.pad_y) / letterbox.scale;
        float x2 = (obj.bbox.x + obj.bbox.width - letterbox.pad_x) / letterbox.scale;
        float y2 = (obj.bbox.y + obj.bbox.height - letterbox.pad_y) / letterbox.scale;
        
        // 将坐标钳制到图像有效范围内 / Clamp coordinates to valid image area
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));
        
        obj.bbox.x = x1;
        obj.bbox.y = y1;
        obj.bbox.width = x2 - x1;
        obj.bbox.height = y2 - y1;
        
        // 同样处理五个关键点坐标 / Apply same mapping to five landmarks
        for (auto& kp : obj.landmarks) {
            float kx = (kp.x - letterbox.pad_x) / letterbox.scale;
            float ky = (kp.y - letterbox.pad_y) / letterbox.scale;
            kp.x = std::max(0.0f, std::min(kx, static_cast<float>(orig_width)));
            kp.y = std::max(0.0f, std::min(ky, static_cast<float>(orig_height)));
        }
    }
}

void CoordinateMapper::MapToOriginalFrame(
    const image_utils::LetterboxInfo& letterbox,
    const CropParams& crop,
    int orig_width, int orig_height,
    std::vector<DetectedObject>* objects) {
    if (!objects) return;
    
    for (auto& obj : *objects) {
        // 第一阶段：将 Letterbox 坐标反算到 ROI 局部坐标系（同单阶段）
        // Stage 1: inverse letterbox to ROI local coordinates (same as 1-stage)
        float x1_roi = (obj.bbox.x - letterbox.pad_x) / letterbox.scale;
        float y1_roi = (obj.bbox.y - letterbox.pad_y) / letterbox.scale;
        float x2_roi = (obj.bbox.x + obj.bbox.width - letterbox.pad_x) / letterbox.scale;
        float y2_roi = (obj.bbox.y + obj.bbox.height - letterbox.pad_y) / letterbox.scale;
        
        // 第二阶段：叠加 ROI 在原图中的偏移量，得到原始全图坐标
        // Stage 2: add ROI offset to obtain coordinates in the original full frame
        float x1 = x1_roi + crop.x;
        float y1 = y1_roi + crop.y;
        float x2 = x2_roi + crop.x;
        float y2 = y2_roi + crop.y;
        
        // 钳制到图像边界 / Clamp to image boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));
        
        obj.bbox.x = x1;
        obj.bbox.y = y1;
        obj.bbox.width = x2 - x1;
        obj.bbox.height = y2 - y1;
        
        // 同样处理关键点坐标 / Apply two-stage mapping to landmarks
        for (auto& kp : obj.landmarks) {
            float kx_roi = (kp.x - letterbox.pad_x) / letterbox.scale;
            float ky_roi = (kp.y - letterbox.pad_y) / letterbox.scale;
            float kx = kx_roi + crop.x;
            float ky = ky_roi + crop.y;
            kp.x = std::max(0.0f, std::min(kx, static_cast<float>(orig_width)));
            kp.y = std::max(0.0f, std::min(ky, static_cast<float>(orig_height)));
        }
    }
}

CropParams CoordinateMapper::ComputeHeadROI(
    const Rect& body_bbox,
    int frame_w, int frame_h,
    const Config::HeadROIParams& params) {
    
    float bx = body_bbox.x;
    float by = body_bbox.y;
    float bw = body_bbox.width;
    float bh = body_bbox.height;
    
    // 若人体框高度小于最小阈值，返回无效 ROI（全零）
    // If body box height is below minimum, return invalid ROI (all zeros)
    if (bh < params.min_body_height) {
        return {0, 0, 0, 0};
    }
    
    float ratio = bh / bw;  // 人体高宽比 / body aspect ratio
    float crop_h_factor = params.crouch_ratio;  // 默认使用蹲姿裁剪比例 / default: crouch
    bool expand_w = false;
    
    /* 根据高宽比自适应选择裁剪策略
     * Adaptive cropping strategy based on body aspect ratio:
     * - ratio >= 2.5 : 全身站立姿态 → 取上半身 + 横向扩展
     *                   full-body standing → upper body + width expansion
     * - ratio >= 1.5 : 半身坐姿 → 中等裁剪比例
     *                   half-body seated → medium crop ratio
     * - else         : 蹲姿/宽体 → 较小裁剪比例
     *                   crouching/wide body → smaller crop ratio
     */
    if (ratio >= 2.5f) {
        crop_h_factor = params.full_body_ratio;
        expand_w = true;
    } else if (ratio >= 1.5f) {
        crop_h_factor = params.half_body_ratio;
    }
    
    float crop_y = by;           // 从人体框顶部开始裁剪 / start from top of body box
    float crop_h = bh * crop_h_factor;  // 裁剪高度 / crop height
    float crop_x = bx;
    float crop_w = bw;
    
    // 对全身姿态，向左右扩展裁剪宽度以包含手臂/肩膀区域
    // For full-body pose, expand crop width to include arms/shoulders
    if (expand_w) {
        float expand_pixels = bw * params.width_expand;
        crop_x = bx - expand_pixels;
        crop_w = bw + 2.0f * expand_pixels;
    }
    
    // 将裁剪框钳制到帧边界内 / Clamp crop to frame boundaries
    int cx = std::max(0, static_cast<int>(crop_x));
    int cy = std::max(0, static_cast<int>(crop_y));
    int cw = static_cast<int>(crop_w);
    int ch = static_cast<int>(crop_h);
    
    if (cx >= frame_w) cx = frame_w - 1;
    if (cy >= frame_h) cy = frame_h - 1;
    if (cx + cw > frame_w) cw = frame_w - cx;
    if (cy + ch > frame_h) ch = frame_h - cy;
    
    return {cx, cy, cw, ch};
}

void CoordinateMapper::NormalizeBBox(Rect& bbox, int width, int height) {
    if (width <= 0 || height <= 0) return;
    // 将像素坐标除以图像宽高映射到 [0, 1]，并钳制边界
    // Map pixel coordinates to [0, 1] by dividing by image dimensions, clamp edges
    bbox.x = std::max(0.0f, std::min(bbox.x / width, 1.0f));
    bbox.y = std::max(0.0f, std::min(bbox.y / height, 1.0f));
    bbox.width = std::max(0.0f, std::min(bbox.width / width, 1.0f - bbox.x));
    bbox.height = std::max(0.0f, std::min(bbox.height / height, 1.0f - bbox.y));
}

} // namespace face_rec
