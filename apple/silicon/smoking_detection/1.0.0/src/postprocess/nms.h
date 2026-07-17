// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - NMS (Non-Maximum Suppression)
// Class-aware NMS for person and cigarette detection.

#pragma once

#include <algorithm>
#include <vector>

#include "../common/types.h"
#include "../common/geometry_utils.h"

namespace smoking {

// NMS 结果
struct NmsResult {
    std::vector<int32_t> kept_indices;
};

// Class-aware NMS
// detections: 已按 confidence 降序排列
// iou_threshold: 重叠超过此阈值时抑制低分框
// 返回保留的索引
inline NmsResult ApplyNMS(const std::vector<Detection>& detections,
                           float iou_threshold) {
    NmsResult result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        result.kept_indices.push_back(static_cast<int32_t>(i));

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;

            float iou = ComputeIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

// 按 confidence 降序排序的辅助
inline void SortByConfidence(std::vector<Detection>& detections) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
}

// 过滤低于置信度阈值的检测
inline std::vector<Detection> FilterByConfidence(
    const std::vector<Detection>& detections, float threshold) {
    std::vector<Detection> result;
    result.reserve(detections.size());
    for (const auto& d : detections) {
        if (d.confidence >= threshold) {
            result.push_back(d);
        }
    }
    return result;
}

// 过滤小于最小尺寸的目标
inline std::vector<Detection> FilterByMinSize(
    const std::vector<Detection>& detections,
    float min_width, float min_height) {
    std::vector<Detection> result;
    result.reserve(detections.size());
    for (const auto& d : detections) {
        if (d.bbox.width >= min_width && d.bbox.height >= min_height) {
            result.push_back(d);
        }
    }
    return result;
}

}  // namespace smoking
