/**
 * tracker.h
 *
 * 目标跟踪模块 - Target Tracking Module
 *
 * 本文件定义了 ByteTracker（基于 IOU 的贪婪匹配跟踪器）的接口与实现。
 * This file defines the interface and implementation of ByteTracker,
 * an IOU-based greedy matching tracker.
 *
 * ByteTracker 的核心思路：
 * 对每一帧的检测结果与现存轨迹计算 IOU，贪心地为每个检测匹配 IOU 最大的轨迹，
 * 未匹配的检测创建新轨迹，未匹配的轨迹递增丢失计数，超限则删除。
 * Core idea: compute IOU between detections and existing tracks, greedily match
 * each detection to the best-overlapping track, create new tracks for unmatched
 * detections, and remove tracks that have been lost for too many frames.
 */

#ifndef FACE_RECOGNITION_TRACKER_H
#define FACE_RECOGNITION_TRACKER_H

#include "common/types.h"
#include <vector>

namespace face_rec {

/**
 * ITracker - 跟踪器抽象接口 / Abstract tracker interface
 *
 * 为不同跟踪算法提供统一的多态基类。
 * Provides a polymorphic base for different tracking algorithms.
 */
class ITracker {
public:
    virtual ~ITracker() = default;

    /** 初始化跟踪器参数 / Initialize tracker parameters */
    virtual void Initialize(float iou_threshold, int max_lost) = 0;

    /**
     * 更新轨迹：输入当前帧检测结果，返回带 track_id 的检测列表
     * Update tracks: accept current-frame detections, return detections with track_id assigned
     */
    virtual std::vector<DetectedObject> Update(const std::vector<DetectedObject>& detections) = 0;

    /** 重置所有轨迹状态 / Reset all track state */
    virtual void Reset() = 0;
};

/**
 * ByteTracker - IOU 贪婪匹配跟踪器 / IOU-based greedy-matching tracker
 *
 * 算法步骤：
 *   1. 将每个检测与已有轨迹计算 IOU，贪心匹配最优轨迹
 *   2. 未匹配的检测作为新目标创建新轨迹
 *   3. 未匹配的旧轨迹丢失计数 +1
 *   4. 移除丢失超过阈值的轨迹
 *
 * Algorithm:
 *   1. Compute IOU between each detection and existing tracks, greedily match
 *   2. Create new tracks for unmatched detections
 *   3. Increment lost count for unmatched old tracks
 *   4. Remove tracks that exceed the max_lost threshold
 */
class ByteTracker : public ITracker {
public:
    ByteTracker() = default;
    ~ByteTracker() override = default;

    void Initialize(float iou_threshold, int max_lost) override;
    std::vector<DetectedObject> Update(const std::vector<DetectedObject>& detections) override;
    void Reset() override;

private:
    /**
     * Track - 单条轨迹 / A single tracked object
     */
    struct Track {
        int id = -1;                    // 轨迹唯一 ID / Unique track ID
        Rect bbox;                      // 当前帧的边界框 / Current-frame bounding box
        float confidence = 0.0f;        // 置信度 / Detection confidence
        int hits = 0;                   // 累计命中帧数 / Consecutive hit frames
        int lost = 0;                   // 连续丢失帧数 / Consecutive lost frames
    };

    std::vector<Track> tracks_;         // 全部活跃轨迹 / All active tracks
    int next_id_ = 1;                   // 下一个轨迹 ID / Next available track ID
    float iou_threshold_ = 0.30f;       // IOU 匹配阈值 / IOU threshold for matching
    int max_lost_ = 30;                 // 最大丢失帧数上限 / Max frames before deleting a lost track

    /**
     * 计算两个边界框的交并比 (IOU)
     * Compute Intersection-over-Union between two bounding boxes
     */
    static float ComputeIOU(const Rect& box1, const Rect& box2);
};

} // namespace face_rec

#endif // FACE_RECOGNITION_TRACKER_H
