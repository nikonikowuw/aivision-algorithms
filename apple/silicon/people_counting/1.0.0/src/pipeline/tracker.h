/**
 * @file tracker.h
 * @brief ByteTrack-based IoU tracker
 */

#ifndef PEOPLE_COUNTING_TRACKER_H
#define PEOPLE_COUNTING_TRACKER_H

#include "common/types.h"
#include <vector>

namespace people_count {

class ITracker {
public:
    virtual ~ITracker() = default;
    virtual std::vector<DetectedObject> Update(const std::vector<DetectedObject>& detections) = 0;
    virtual void Reset() = 0;
};

class ByteTracker : public ITracker {
public:
    struct Track {
        int id;
        Rect bbox;
        float confidence;
        int frames_since_update;
    };

    explicit ByteTracker(float iou_threshold = 0.2f, int max_lost_frames = 30);
    ~ByteTracker() override;

    std::vector<DetectedObject> Update(const std::vector<DetectedObject>& detections) override;
    void Reset() override;

private:
    float iou_threshold_;
    int max_lost_frames_;
    std::vector<Track> active_tracks_;
    int next_track_id_ = 0;

    std::vector<int> MatchIoU(const std::vector<DetectedObject>& detections,
                              std::vector<Track>& tracks);
};

} // namespace people_count

#endif // PEOPLE_COUNTING_TRACKER_H
