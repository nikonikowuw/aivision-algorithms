// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Person Tracker Implementation

#include "tracker.h"

#include <algorithm>
#include <limits>

#include "../common/geometry_utils.h"

namespace smoking {

PersonTracker::PersonTracker(const TrackerConfig& config)
    : config_(config) {}

std::vector<TrackedPerson> PersonTracker::Update(
    const std::vector<Detection>& detections,
    int64_t current_time_ms) {

    std::vector<TrackedPerson> result;
    MatchDetections(detections, current_time_ms, result);
    CleanupExpiredTracks(current_time_ms);
    return result;
}

std::vector<TrackState> PersonTracker::GetActiveTracks() const {
    std::vector<TrackState> active;
    for (const auto& [id, state] : tracks_) {
        if (state.is_active) {
            active.push_back(state);
        }
    }
    return active;
}

void PersonTracker::Reset() {
    tracks_.clear();
    next_id_ = 0;
}

void PersonTracker::MatchDetections(
    const std::vector<Detection>& detections,
    int64_t current_time_ms,
    std::vector<TrackedPerson>& result) {

    // 构建活跃轨迹列表
    std::vector<int32_t> active_ids;
    for (auto& [id, state] : tracks_) {
        if (state.is_active) {
            active_ids.push_back(id);
        }
    }

    // 标记已匹配的检测和轨迹
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(active_ids.size(), false);

    // 第一轮：按 confidence 降序匹配
    for (size_t di = 0; di < detections.size(); ++di) {
        float best_iou = config_.iou_threshold;
        float best_center_dist = config_.center_dist_max;
        int best_ti = -1;

        for (size_t ti = 0; ti < active_ids.size(); ++ti) {
            if (track_matched[ti]) continue;

            const auto& track = tracks_[active_ids[ti]];
            float iou = ComputeIoU(detections[di].bbox, track.bbox);
            float center_dist = CenterDistanceNormalized(
                detections[di].bbox, track.bbox);

            // IoU 匹配优先，中心距离补偿
            if (iou > best_iou ||
                (iou > 0.01f && center_dist < best_center_dist)) {
                best_iou = iou;
                best_center_dist = center_dist;
                best_ti = static_cast<int>(ti);
            }
        }

        if (best_ti >= 0) {
            // 匹配成功，更新轨迹
            int32_t tid = active_ids[best_ti];
            tracks_[tid].bbox = detections[di].bbox;
            tracks_[tid].confidence = detections[di].confidence;
            tracks_[tid].last_update_ms = current_time_ms;
            tracks_[tid].frames_since_update = 0;
            tracks_[tid].is_active = true;

            det_matched[di] = true;
            track_matched[best_ti] = true;

            TrackedPerson tp;
            tp.track_id = tid;
            tp.bbox = detections[di].bbox;
            tp.confidence = detections[di].confidence;
            tp.last_update_ms = current_time_ms;
            tp.first_seen_ms = tracks_[tid].first_seen_ms;
            result.push_back(tp);
        }
    }

    // 新轨迹必须在创建它的同一分析周期参与 ROI 和事件处理。
    for (size_t di = 0; di < detections.size(); ++di) {
        if (!det_matched[di]) {
            result.push_back(CreateTrack(detections[di], current_time_ms));
        }
    }

    // 未匹配的活跃轨迹标记为丢失
    for (size_t ti = 0; ti < active_ids.size(); ++ti) {
        if (!track_matched[ti]) {
            tracks_[active_ids[ti]].frames_since_update++;
            int64_t elapsed = current_time_ms - tracks_[active_ids[ti]].last_update_ms;
            if (elapsed > config_.max_lost_ms ||
                tracks_[active_ids[ti]].frames_since_update > config_.max_lost_frames) {
                tracks_[active_ids[ti]].is_active = false;
            }
        }
    }
}

TrackedPerson PersonTracker::CreateTrack(const Detection& det,
                                         int64_t current_time_ms) {
    TrackState state;
    state.track_id = next_id_++;
    state.bbox = det.bbox;
    state.confidence = det.confidence;
    state.last_update_ms = current_time_ms;
    state.first_seen_ms = current_time_ms;
    state.frames_since_update = 0;
    state.is_active = true;

    tracks_[state.track_id] = state;
    return {state.track_id, state.bbox, state.confidence,
            state.last_update_ms, state.first_seen_ms};
}

void PersonTracker::CleanupExpiredTracks(int64_t current_time_ms) {
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (!it->second.is_active) {
            int64_t elapsed = current_time_ms - it->second.last_update_ms;
            if (elapsed > config_.max_lost_ms * 2) {
                it = tracks_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

}  // namespace smoking
