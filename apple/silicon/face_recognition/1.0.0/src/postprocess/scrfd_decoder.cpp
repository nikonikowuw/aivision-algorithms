/**
 * @file scrfd_decoder.cpp
 * @brief SCRFD 检测器解码实现
 *        Implementations of SCRFD detection decoding
 *
 * 实现基于 Anchor 的 SCRFD 解码逻辑：对每个 stride 层级的每个网格点的每个 Anchor，
 * 解析分数、边框偏移和关键点偏移，再通过 NMS 去除冗余检测。
 * Implements anchor-based SCRFD decoding: for each anchor at each grid cell
 * across stride levels, parse scores, bbox offsets, and landmark offsets,
 * then apply NMS to eliminate redundant detections.
 */
#include "scrfd_decoder.h"
#include <algorithm>
#include <cmath>

namespace face_rec {

/**
 * @brief 计算两个矩形框的交并比（IoU）
 *        Compute Intersection over Union (IoU) between two rectangles
 * @param box1 矩形框 1 / box 1
 * @param box2 矩形框 2 / box 2
 * @return IoU 值（[0, 1] 范围） / IoU value in [0, 1]
 */
static float ComputeIoU(const Rect& box1, const Rect& box2) {
    // 计算交集区域的左上角和右下角坐标
    // Compute intersection top-left and bottom-right coordinates
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    float intersection_w = std::max(0.0f, x2 - x1);
    float intersection_h = std::max(0.0f, y2 - y1);
    float intersection_area = intersection_w * intersection_h;
    
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection_area;
    
    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

std::vector<DetectedObject> ScrfdDecoder::Decode(
    const std::vector<StrideOutput>& outputs,
    float conf_threshold,
    float iou_threshold,
    int max_count) {
    
    std::vector<DetectedObject> candidates;
    
    // 遍历各个 stride 层级的特征图
    // Iterate over each stride-level feature map
    for (const auto& out : outputs) {
        int stride = out.stride;
        int grid_h = out.grid_h;
        int grid_w = out.grid_w;
        int anchor_count = out.anchor_count;
        
        // 遍历特征图上每个网格点 / Iterate over each grid cell
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                // 遍历每个 Anchor / Iterate over each anchor
                for (int k = 0; k < anchor_count; ++k) {
                    int idx = (y * grid_w + x) * anchor_count + k;
                    float score = out.scores[idx];
                    
                    // 过滤低分检测 / Filter low-confidence detections
                    if (score < conf_threshold) continue;
                    
                    // 网格中心在原图中的像素坐标
                    // Pixel coordinate of grid cell center in the original image
                    float cx = x * stride;
                    float cy = y * stride;
                    
                    /* SCRFD 边框解码：模型输出的是相对网格中心的偏移（以 stride 为单位）
                     * SCRFD bbox decoding: model outputs offsets relative to grid
                     * center, normalized by stride
                     *   x1 = cx - dx1 * stride
                     *   y1 = cy - dy1 * stride
                     *   x2 = cx + dx2 * stride
                     *   y2 = cy + dy2 * stride
                     */
                    float dx1 = out.bboxes[idx * 4 + 0];
                    float dy1 = out.bboxes[idx * 4 + 1];
                    float dx2 = out.bboxes[idx * 4 + 2];
                    float dy2 = out.bboxes[idx * 4 + 3];
                    
                    float x1 = cx - dx1 * stride;
                    float y1 = cy - dy1 * stride;
                    float x2 = cx + dx2 * stride;
                    float y2 = cy + dy2 * stride;
                    
                    DetectedObject obj;
                    obj.bbox = Rect{x1, y1, x2 - x1, y2 - y1};
                    obj.confidence = score;
                    obj.label = 2; // 默认标签（人脸） / default label (face)
                    
                    // 若模型输出包含关键点数据，解码 5 个关键点（每点 x, y 偏移）
                    // If model outputs landmarks, decode 5 keypoints (each has x, y offset)
                    if (out.kps) {
                        for (int p = 0; p < 5; ++p) {
                            float dkx = out.kps[idx * 10 + p * 2 + 0];
                            float dky = out.kps[idx * 10 + p * 2 + 1];
                            obj.landmarks[p].x = cx + dkx * stride;
                            obj.landmarks[p].y = cy + dky * stride;
                        }
                    }
                    
                    candidates.push_back(obj);
                }
            }
        }
    }
    
    return NMS(candidates, iou_threshold, max_count);
}

std::vector<DetectedObject> ScrfdDecoder::NMS(
    const std::vector<DetectedObject>& detections,
    float iou_threshold,
    int max_count) {
    
    std::vector<DetectedObject> results;
    if (detections.empty()) return results;
    
    /* 标准贪心 NMS（Greedy NMS）算法
     * Standard greedy NMS algorithm:
     * 1. 按置信度降序排列 / Sort by confidence descending
     * 2. 保留最高分检测框 / Keep the highest-scoring box
     * 3. 移除与该框 IoU 超过阈值的其他框 / Remove boxes with IoU above threshold
     * 4. 重复直到处理完所有框或达到 max_count / Repeat until all processed or max_count
     */
    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        indices[i] = i;
    }
    
    // 按置信度降序排列索引 / Sort indices by confidence descending
    std::sort(indices.begin(), indices.end(), [&detections](int i1, int i2) {
        return detections[i1].confidence > detections[i2].confidence;
    });
    
    std::vector<bool> active(detections.size(), true);
    int count = 0;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (!active[idx]) continue;
        
        results.push_back(detections[idx]);
        if (max_count > 0 && ++count >= max_count) break;
        
        // 抑制与当前框高度重叠的其他检测框
        // Suppress other detections with high overlap to the current box
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int target_idx = indices[j];
            if (!active[target_idx]) continue;
            
            float iou = ComputeIoU(detections[idx].bbox, detections[target_idx].bbox);
            if (iou >= iou_threshold) {
                active[target_idx] = false;
            }
        }
    }
    
    return results;
}

} // namespace face_rec
