// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - DAMO-YOLO Decoder Implementation

#include "damoyolo_decoder.h"

#include <algorithm>
#include <cmath>

#include "../common/logger.h"
#include "nms.h"

namespace smoking {

std::vector<Detection> DamoOutputParser::Parse(
    const float* raw_output,
    int32_t num_detections,
    int32_t /*model_w*/,
    int32_t /*model_h*/,
    const ImageTransform& transform) {

    std::vector<Detection> results;
    if (!raw_output || num_detections <= 0 || transform.scale <= 0.0f ||
        transform.frame_width <= 0 || transform.frame_height <= 0) {
        return results;
    }

    results.reserve(num_detections);

    for (int32_t i = 0; i < num_detections; ++i) {
        const float* det = raw_output + i * kFieldsPerDetection;

        float x1 = det[0];  // 模型坐标系左上角
        float y1 = det[1];
        float x2 = det[2];  // 模型坐标系右下角
        float y2 = det[3];
        float score = det[4];
        float class_id = det[5];

        // 基本有效性检查
        if (!IsFinite(score) || !IsFinite(class_id) || !IsFinite(x1) ||
            !IsFinite(y1) || !IsFinite(x2) || !IsFinite(y2) ||
            score < 0.0f || score > 1.0f) {
            continue;
        }

        // 转换为模型坐标系的 RectF
        RectF model_rect;
        model_rect.x = std::min(x1, x2);
        model_rect.y = std::min(y1, y2);
        model_rect.width = std::abs(x2 - x1);
        model_rect.height = std::abs(y2 - y1);

        // 通过 ImageTransform 映射回原图像素坐标
        RectF original_rect = transform.ModelToOriginal(model_rect);

        // 裁剪到原图边界
        original_rect.ClipToFrame(
            static_cast<float>(transform.frame_width),
            static_cast<float>(transform.frame_height));

        // 二次检查映射后的坐标
        if (!IsValidRect(original_rect) || original_rect.width <= 0.0f ||
            original_rect.height <= 0.0f) {
            continue;
        }

        Detection det_result;
        det_result.bbox = original_rect;
        det_result.confidence = score;
        det_result.class_id = static_cast<int32_t>(class_id);
        results.push_back(det_result);
    }

    // 按 confidence 降序排序
    SortByConfidence(results);

    return results;
}

}  // namespace smoking
