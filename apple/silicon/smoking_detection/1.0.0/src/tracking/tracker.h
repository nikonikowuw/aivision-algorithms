// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Person Tracker
// ByteTrack 风格的检测关联，基于 IoU + 中心距离补偿。

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../common/types.h"

namespace smoking {

// 轨迹状态
struct TrackState {
    int32_t track_id       = -1;
    RectF   bbox;
    float   confidence     = 0.0f;
    int64_t last_update_ms = 0;
    int64_t first_seen_ms  = 0;
    int32_t frames_since_update = 0;
    bool    is_active      = true;
};

// 跟踪器配置
struct TrackerConfig {
    float   iou_threshold    = 0.20f;  // IoU 匹配阈值
    float   center_dist_max  = 0.5f;   // 中心距离补偿最大归一化距离
    int64_t max_lost_ms      = 2000;   // 最大丢失时间（ms）
    int32_t max_lost_frames  = 15;     // 最大丢失帧数
};

// ITracker — 跟踪器接口
class ITracker {
public:
    virtual ~ITracker() = default;

    // 输入检测列表（原图像素坐标），返回更新后的跟踪列表
    virtual std::vector<TrackedPerson> Update(
        const std::vector<Detection>& detections,
        int64_t current_time_ms) = 0;

    // 获取所有活跃轨迹
    virtual std::vector<TrackState> GetActiveTracks() const = 0;

    // 重置
    virtual void Reset() = 0;
};

// PersonTracker — ByteTrack 风格的简化实现
class PersonTracker : public ITracker {
public:
    explicit PersonTracker(const TrackerConfig& config = TrackerConfig{});

    std::vector<TrackedPerson> Update(
        const std::vector<Detection>& detections,
        int64_t current_time_ms) override;

    std::vector<TrackState> GetActiveTracks() const override;

    void Reset() override;

private:
    TrackerConfig config_;
    std::unordered_map<int32_t, TrackState> tracks_;
    int32_t next_id_ = 0;

    // 匹配检测到现有轨迹
    void MatchDetections(
        const std::vector<Detection>& detections,
        int64_t current_time_ms,
        std::vector<TrackedPerson>& result);

    // 创建新轨迹并返回本次可见结果
    TrackedPerson CreateTrack(const Detection& det, int64_t current_time_ms);

    // 清理过期轨迹
    void CleanupExpiredTracks(int64_t current_time_ms);
};

}  // namespace smoking
