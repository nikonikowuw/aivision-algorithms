/**
 * @file tracker.h
 * @brief GPU Face Recognition — ByteTrack multi-object tracker.
 *        IoU-based association with face feature ID switch correction.
 * @module Pipeline Layer
 */

#ifndef GPU_FACE_RECOGNITION_TRACKER_H
#define GPU_FACE_RECOGNITION_TRACKER_H

#include "common/types.h"
#include <vector>
#include <unordered_map>

namespace face_rec {

/**
 * @class ITracker
 * @brief Abstract interface for multi-object tracking.
 */
class ITracker {
public:
    virtual ~ITracker() = default;

    /**
     * @brief Update tracker with new detections.
     * @param detections Current frame detections
     * @return Tracked objects with stable track_ids
     */
    virtual std::vector<DetectedObject> Update(
        const std::vector<DetectedObject>& detections) = 0;

    /** @brief Reset tracker state (e.g. on stream reconnect) */
    virtual void Reset() = 0;
};

/**
 * @class ByteTracker
 * @brief ByteTrack-inspired IoU-based tracker.
 *        Uses two-stage matching: high-confidence + low-confidence detections.
 *        Designed for 1-2 FPS streams with lower IoU thresholds (0.1-0.2).
 */
class ByteTracker : public ITracker {
public:
    /**
     * @brief Constructor
     * @param iou_threshold IoU matching threshold (default 0.2 for low FPS)
     * @param max_lost_frames Maximum frames before a track is removed
     */
    explicit ByteTracker(float iou_threshold = 0.2f, int max_lost_frames = 30);
    ~ByteTracker() override;

    std::vector<DetectedObject> Update(
        const std::vector<DetectedObject>& detections) override;

    void Reset() override;

private:
    struct Track {
        int id;
        Rect bbox;
        float confidence;
        int frames_since_update;
        std::vector<float> embedding;  // Latest face embedding for ID correction
    };

    float iou_threshold_;
    int max_lost_frames_;
    std::vector<Track> active_tracks_;
    int next_track_id_ = 0;

    std::vector<int> MatchIoU(const std::vector<DetectedObject>& detections,
                              std::vector<Track>& tracks);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_TRACKER_H
