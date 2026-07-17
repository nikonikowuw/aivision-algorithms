// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Event State Machine Implementation

#include "event_state.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

#include "../common/geometry_utils.h"

namespace smoking {
namespace {

int32_t CountHits(const std::vector<bool>& history) {
    return static_cast<int32_t>(
        std::count(history.begin(), history.end(), true));
}

SmokingEvent EmptyEvent() {
    SmokingEvent event;
    event.track_id = -1;
    return event;
}

}  // namespace

EventStateMachine::EventStateMachine(const Config& config) : config_(config) {}

SmokingEvent EventStateMachine::ProcessTrack(
    int32_t track_id, bool has_evidence, const RectF& person_bbox,
    const RectF& cigarette_bbox, float confidence, int64_t current_time_ms) {
    if (track_id < 0) return EmptyEvent();

    auto it = track_states_.find(track_id);
    if (it == track_states_.end()) {
        TrackEventState state;
        if (!TryInheritFromTombstone(person_bbox, current_time_ms, state)) {
            state.instance_nonce = next_instance_nonce_++;
        }
        it = track_states_.emplace(track_id, std::move(state)).first;
    }

    auto& state = it->second;
    state.last_person_bbox = person_bbox;
    if (has_evidence) {
        state.last_cigarette_bbox = cigarette_bbox;
        state.last_confidence = confidence;
    }

    if (state.state == EventState::kConfirmedLatched) {
        if (has_evidence) {
            state.last_hit_ms = current_time_ms;
        } else if (current_time_ms - state.last_hit_ms >= config_.rearm_ms) {
            state = TrackEventState{};
            state.instance_nonce = next_instance_nonce_++;
            state.last_person_bbox = person_bbox;
        }
        return EmptyEvent();
    }

    if (state.state == EventState::kRearmPending) {
        state.state = EventState::kIdle;
        state.history.clear();
        state.event_outputted = false;
    }

    if (state.state == EventState::kIdle) {
        if (!has_evidence) return EmptyEvent();
        state.state = EventState::kCandidate;
        state.event_start_ms = current_time_ms;
        state.last_hit_ms = current_time_ms;
        state.history.clear();
    }

    state.history.push_back(has_evidence);
    while (static_cast<int32_t>(state.history.size()) > config_.temporal_window) {
        state.history.erase(state.history.begin());
    }
    if (has_evidence) state.last_hit_ms = current_time_ms;

    if (CountHits(state.history) >= config_.confirm_hits) {
        return Confirm(track_id, state, person_bbox, cigarette_bbox, confidence);
    }

    if (!has_evidence && current_time_ms - state.last_hit_ms >= config_.rearm_ms) {
        state = TrackEventState{};
        state.instance_nonce = next_instance_nonce_++;
        state.last_person_bbox = person_bbox;
    }
    return EmptyEvent();
}

void EventStateMachine::RetireMissingTracks(
    const std::vector<int32_t>& active_track_ids, int64_t current_time_ms) {
    const std::unordered_set<int32_t> active(active_track_ids.begin(),
                                              active_track_ids.end());
    for (auto it = track_states_.begin(); it != track_states_.end();) {
        if (active.count(it->first) != 0) {
            ++it;
            continue;
        }
        if (it->second.state == EventState::kCandidate ||
            it->second.state == EventState::kConfirmedLatched) {
            CreateTombstone(it->second, current_time_ms);
        }
        it = track_states_.erase(it);
    }
}

void EventStateMachine::CleanupTombstones(int64_t current_time_ms) {
    tombstones_.erase(
        std::remove_if(tombstones_.begin(), tombstones_.end(),
                       [current_time_ms](const EventTombstone& tombstone) {
                           return current_time_ms > tombstone.expire_ms;
                       }),
        tombstones_.end());
}

const TrackEventState* EventStateMachine::GetTrackState(int32_t track_id) const {
    const auto it = track_states_.find(track_id);
    return it == track_states_.end() ? nullptr : &it->second;
}

void EventStateMachine::Reset() {
    track_states_.clear();
    tombstones_.clear();
    next_instance_nonce_ = 1;
    next_event_sequence_ = 1;
}

bool EventStateMachine::TryInheritFromTombstone(
    const RectF& person_bbox, int64_t current_time_ms,
    TrackEventState& inherited) {
    size_t best_index = tombstones_.size();
    float best_iou = config_.tombstone_match_iou;
    for (size_t i = 0; i < tombstones_.size(); ++i) {
        if (current_time_ms > tombstones_[i].expire_ms) continue;
        const float iou = ComputeIoU(person_bbox,
                                     tombstones_[i].state.last_person_bbox);
        if (iou >= best_iou) {
            best_iou = iou;
            best_index = i;
        }
    }
    if (best_index == tombstones_.size()) return false;

    inherited = tombstones_[best_index].state;
    inherited.last_person_bbox = person_bbox;
    tombstones_.erase(tombstones_.begin() + static_cast<std::ptrdiff_t>(best_index));
    return true;
}

SmokingEvent EventStateMachine::Confirm(
    int32_t track_id, TrackEventState& state, const RectF& person_bbox,
    const RectF& cigarette_bbox, float confidence) {
    if (state.event_outputted) return EmptyEvent();

    state.state = EventState::kConfirmedLatched;
    state.event_outputted = true;
    state.event_sequence = next_event_sequence_++;

    SmokingEvent event;
    event.category_code = 14001;
    event.detect_confidence = confidence;
    event.person_bbox = person_bbox;
    event.cigarette_bbox = cigarette_bbox;
    event.track_id = track_id;
    event.event_id = GenerateEventId(state.instance_nonce, state.event_sequence);
    return event;
}

std::string EventStateMachine::GenerateEventId(int32_t instance_nonce,
                                               int32_t sequence) const {
    std::ostringstream stream;
    stream << "smoking:" << std::hex << instance_nonce << ":" << std::dec
           << sequence;
    return stream.str();
}

void EventStateMachine::CreateTombstone(const TrackEventState& state,
                                        int64_t current_time_ms) {
    EventTombstone tombstone;
    tombstone.state = state;
    tombstone.expire_ms = current_time_ms + config_.rearm_ms;
    tombstones_.push_back(std::move(tombstone));
}

}  // namespace smoking
