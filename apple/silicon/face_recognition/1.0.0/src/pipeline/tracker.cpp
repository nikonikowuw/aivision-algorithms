/**
 * tracker.cpp
 *
 * 目标跟踪模块实现 - Target Tracking Module Implementation
 *
 * 实现了 ByteTracker 的完整 IOU 贪婪匹配逻辑。
 * Implements the complete IOU greedy-matching logic of ByteTracker.
 */

#include "tracker.h"
#include <algorithm>

namespace face_rec {

/**
 * 初始化跟踪器参数并重置状态
 * Initialize tracker parameters and reset internal state
 */
void ByteTracker::Initialize(float iou_threshold, int max_lost) {
    iou_threshold_ = iou_threshold;
    max_lost_ = max_lost;
    Reset();
}

/**
 * 重置所有轨迹，清空 ID 计数器
 * Clear all tracks and reset the ID counter
 */
void ByteTracker::Reset() {
    tracks_.clear();
    next_id_ = 1;
}

/**
 * 计算两个矩形框的交并比 (IOU)
 * Compute Intersection-over-Union between two rectangles
 *
 * IOU = Intersection(A, B) / Union(A, B)
 * 当不相交时返回 0.0f / Returns 0.0f when there is no overlap
 */
float ByteTracker::ComputeIOU(const Rect& box1, const Rect& box2) {
    // 计算交集区域的左上角和右下角 / Compute intersection rect
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    // 不相交，交集面积为 0 / No overlap
    if (x1 >= x2 || y1 >= y2) return 0.0f;

    // 交集面积 / Intersection area
    float intersection = (x2 - x1) * (y2 - y1);
    // 各自面积 / Individual areas
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;

    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

/**
 * 核心跟踪算法：4 步贪心 IOU 匹配
 * Core tracking algorithm: 4-step greedy IOU matching
 *
 * Step 1: 将当前帧检测与已有轨迹匹配（贪心 IOU 最大化）
 *         Match current-frame detections with existing tracks (greedy max IOU)
 * Step 2: 未匹配的检测创建新轨迹
 *         Create new tracks for unmatched detections
 * Step 3: 未匹配的旧轨迹丢失计数递增
 *         Increment lost counter for unmatched old tracks
 * Step 4: 移除丢失过久的轨迹
 *         Remove tracks that have been lost beyond max_lost_
 */
std::vector<DetectedObject> ByteTracker::Update(const std::vector<DetectedObject>& detections) {
    // 拷贝输入检测，将在其上标记 track_id
    std::vector<DetectedObject> tracked_detections = detections;
    std::vector<bool> track_matched(tracks_.size(), false);  // 轨迹是否已匹配 / track matched flag
    std::vector<bool> det_matched(detections.size(), false); // 检测是否已匹配 / detection matched flag

    // ---- Step 1: 贪心 IOU 匹配 ----
    // Greedy IOU matching: each detection finds the best-overlapping unmatched track
    for (size_t i = 0; i < tracked_detections.size(); ++i) {
        float max_iou = 0.0f;
        int best_track_idx = -1;

        // 遍历所有未匹配的轨迹，找 IOU 最大的那个
        for (size_t j = 0; j < tracks_.size(); ++j) {
            if (track_matched[j]) continue;
            float iou = ComputeIOU(tracked_detections[i].bbox, tracks_[j].bbox);
            if (iou > max_iou) {
                max_iou = iou;
                best_track_idx = static_cast<int>(j);
            }
        }

        // IOU 超过阈值则匹配成功 / Match successful if IOU >= threshold
        if (best_track_idx != -1 && max_iou >= iou_threshold_) {
            // 用检测更新轨迹的 bbox 和置信度 / Update track with detection
            tracks_[best_track_idx].bbox = tracked_detections[i].bbox;
            tracks_[best_track_idx].confidence = tracked_detections[i].confidence;
            tracks_[best_track_idx].lost = 0;       // 重置丢失计数 / Reset lost counter
            tracks_[best_track_idx].hits++;          // 命中计数 +1 / Increment hit counter

            tracked_detections[i].track_id = tracks_[best_track_idx].id; // 分配轨迹 ID
            track_matched[best_track_idx] = true;
            det_matched[i] = true;
        }
    }

    // ---- Step 2: 未匹配的检测 → 创建新轨迹 ----
    // Unmatched detections become new tracks
    for (size_t i = 0; i < tracked_detections.size(); ++i) {
        if (det_matched[i]) continue;

        Track new_track;
        new_track.id = next_id_++;
        new_track.bbox = tracked_detections[i].bbox;
        new_track.confidence = tracked_detections[i].confidence;
        new_track.hits = 1;
        new_track.lost = 0;

        tracks_.push_back(new_track);
        tracked_detections[i].track_id = new_track.id;
    }

    // ---- Step 3: 未匹配的旧轨迹 → 丢失计数递增 ----
    // Unmatched old tracks increment their lost counter
    // Note: Use track_matched.size() (tracks_ size BEFORE Step 2 added new tracks)
    for (size_t j = 0; j < track_matched.size(); ++j) {
        if (track_matched[j]) continue;
        tracks_[j].lost++;
    }

    // ---- Step 4: 移除丢失超限的轨迹 ----
    // Remove tracks whose lost count exceeds max_lost_
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) { return t.lost > max_lost_; }),
        tracks_.end());

    return tracked_detections;
}

} // namespace face_rec
