// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Event State Machine
// Each person track owns temporal evidence and a one-shot event latch.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/types.h"

namespace smoking {

enum class EventState : int32_t {
    kIdle = 0,
    kCandidate,
    kConfirmedLatched,
    kRearmPending,
};

struct TrackEventState {
    EventState state = EventState::kIdle;
    std::vector<bool> history;
    int64_t event_start_ms = 0;
    int64_t last_hit_ms = 0;
    int32_t event_sequence = 0;
    bool event_outputted = false;
    RectF last_person_bbox;
    RectF last_cigarette_bbox;
    float last_confidence = 0.0f;
    int32_t instance_nonce = 0;
};

struct EventTombstone {
    TrackEventState state;
    int64_t expire_ms = 0;
};

class EventStateMachine {
public:
    struct Config {
        int32_t temporal_window = 5;
        int32_t confirm_hits = 3;
        int64_t rearm_ms = 2000;
        float tombstone_match_iou = 0.3f;
    };

    explicit EventStateMachine(const Config& config);

    /**
     * @brief Record one completed cigarette evaluation for a person track.
     * @return A populated event only on the transition into confirmed state.
     */
    SmokingEvent ProcessTrack(int32_t track_id, bool has_evidence,
                              const RectF& person_bbox,
                              const RectF& cigarette_bbox, float confidence,
                              int64_t current_time_ms);

    /**
     * @brief Retire states whose person tracks disappeared and preserve one
     * short-lived candidate or confirmed lock for spatial inheritance.
     */
    void RetireMissingTracks(const std::vector<int32_t>& active_track_ids,
                             int64_t current_time_ms);

    void CleanupTombstones(int64_t current_time_ms);
    const TrackEventState* GetTrackState(int32_t track_id) const;
    void Reset();

private:
    Config config_;
    std::unordered_map<int32_t, TrackEventState> track_states_;
    std::vector<EventTombstone> tombstones_;
    int32_t next_instance_nonce_ = 1;
    int32_t next_event_sequence_ = 1;

    bool TryInheritFromTombstone(const RectF& person_bbox,
                                 int64_t current_time_ms,
                                 TrackEventState& inherited);
    SmokingEvent Confirm(int32_t track_id, TrackEventState& state,
                         const RectF& person_bbox,
                         const RectF& cigarette_bbox, float confidence);
    std::string GenerateEventId(int32_t instance_nonce, int32_t sequence) const;
    void CreateTombstone(const TrackEventState& state, int64_t current_time_ms);
};

}  // namespace smoking
