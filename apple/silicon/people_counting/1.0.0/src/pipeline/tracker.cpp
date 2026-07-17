/**
 * @file tracker.cpp
 * @brief ByteTracker implementation
 */

#include "tracker.h"
#include "common/geometry_utils.h"
#include "common/logger.h"
#include <algorithm>

namespace people_count {

ByteTracker::ByteTracker(float iou_threshold, int max_lost_frames)
    : iou_threshold_(iou_threshold), max_lost_frames_(max_lost_frames) {}

ByteTracker::~ByteTracker() = default;

void ByteTracker::Reset() {
    active_tracks_.clear();
    next_track_id_ = 0;
}

std::vector<int> ByteTracker::MatchIoU(const std::vector<DetectedObject>& detections,
                                      std::vector<Track>& tracks) {
    std::vector<int> match_dets(detections.size(), -1);
    std::vector<bool> track_matched(tracks.size(), false);

    for (size_t d = 0; d < detections.size(); ++d) {
        float best_iou = iou_threshold_;
        int best_track = -1;

        for (size_t t = 0; t < tracks.size(); ++t) {
            if (track_matched[t]) continue;
            float iou = ComputeIoU(detections[d].bbox, tracks[t].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_track = static_cast<int>(t);
            }
        }

        if (best_track >= 0) {
            match_dets[d] = best_track;
            track_matched[best_track] = true;
        }
    }

    return match_dets;
}

std::vector<DetectedObject> ByteTracker::Update(const std::vector<DetectedObject>& detections) {
    std::vector<DetectedObject> results;

    if (detections.empty()) {
        for (auto& track : active_tracks_) {
            track.frames_since_update++;
        }
    } else {
        auto matches = MatchIoU(detections, active_tracks_);

        for (size_t d = 0; d < detections.size(); ++d) {
            DetectedObject result = detections[d];

            if (matches[d] >= 0) {
                auto& track = active_tracks_[matches[d]];
                track.bbox = detections[d].bbox;
                track.confidence = detections[d].confidence;
                track.frames_since_update = 0;
                result.track_id = track.id;
            } else {
                Track new_track;
                new_track.id = next_track_id_++;
                new_track.bbox = detections[d].bbox;
                new_track.confidence = detections[d].confidence;
                new_track.frames_since_update = 0;
                active_tracks_.push_back(new_track);
                result.track_id = new_track.id;
            }

            results.push_back(result);
        }
    }

    active_tracks_.erase(
        std::remove_if(active_tracks_.begin(), active_tracks_.end(),
                       [this](const Track& t) {
                           return t.frames_since_update > max_lost_frames_;
                       }),
        active_tracks_.end());

    return results;
}

} // namespace people_count
