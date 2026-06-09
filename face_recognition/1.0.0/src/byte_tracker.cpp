#include "face_recognition.h"
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace face_recognition {

// 简化的ByteTracker实现
class ByteTracker : public ObjectTracker {
public:
    ByteTracker() = default;
    ~ByteTracker() override = default;
    
    bool Initialize() override {
        tracks_.clear();
        next_id_ = 1;
        frame_count_ = 0;
        return true;
    }
    
    std::vector<PersonDetection> Update(
        const std::vector<PersonDetection>& detections,
        const cv::Mat& image) override {
        
        frame_count_++;
        for (auto& track : tracks_) Predict(track);
        
        auto matches = MatchDetections(detections);
        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> track_matched(tracks_.size(), false);
        
        for (const auto& match : matches) {
            int di = match.first, ti = match.second;
            tracks_[ti].bbox = detections[di].bbox;
            tracks_[ti].confidence = detections[di].confidence;
            tracks_[ti].hits++;
            tracks_[ti].missed = 0;
            tracks_[ti].state = TrackState::Tracked;
            det_matched[di] = true;
            track_matched[ti] = true;
        }
        
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (!track_matched[i]) {
                if (++tracks_[i].missed > max_age_) tracks_[i].state = TrackState::Lost;
            }
        }
        
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!det_matched[i]) {
                tracks_.push_back({next_id_++, detections[i].bbox, detections[i].confidence, 1, 1, 0, TrackState::Tracked, {0,0}});
            }
        }
        
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& t){ return t.state == TrackState::Lost; }), tracks_.end());
        
        std::vector<PersonDetection> results;
        for (const auto& t : tracks_) {
            if (t.state == TrackState::Tracked) results.push_back({t.bbox, t.confidence, t.id});
        }
        return results;
    }
    
    void Reset() override {
        tracks_.clear();
        next_id_ = 1;
        frame_count_ = 0;
    }
    
private:
    enum class TrackState {
        Tracked,
        Lost
    };
    
    struct Track {
        int id;
        cv::Rect bbox;
        float confidence;
        int age;
        int hits;
        int missed;
        TrackState state;
        cv::Point2f velocity;
    };
    
    void Predict(Track& track) {
        // 简单的卡尔曼滤波预测
        // 这里使用简单的线性预测
        track.velocity = cv::Point2f(
            static_cast<float>(track.bbox.x) / track.age,
            static_cast<float>(track.bbox.y) / track.age
        );
        
        track.bbox.x += static_cast<int>(track.velocity.x);
        track.bbox.y += static_cast<int>(track.velocity.y);
        track.age++;
    }
    
    std::vector<std::pair<int, int>> MatchDetections(
        const std::vector<PersonDetection>& detections) {
        
        std::vector<std::pair<int, int>> matches;
        
        if (detections.empty() || tracks_.empty()) {
            return matches;
        }
        
        // 计算IOU矩阵
        std::vector<std::vector<float>> iou_matrix(
            detections.size(), std::vector<float>(tracks_.size()));
        
        for (size_t i = 0; i < detections.size(); ++i) {
            for (size_t j = 0; j < tracks_.size(); ++j) {
                if (tracks_[j].state == TrackState::Tracked) {
                    iou_matrix[i][j] = CalculateIOU(detections[i].bbox, tracks_[j].bbox);
                } else {
                    iou_matrix[i][j] = 0.0f;
                }
            }
        }
        
        // 贪心匹配
        std::vector<bool> det_matched(detections.size(), false);
        std::vector<bool> track_matched(tracks_.size(), false);
        
        while (true) {
            float max_iou = 0.0f;
            int best_det = -1;
            int best_track = -1;
            
            for (size_t i = 0; i < detections.size(); ++i) {
                if (det_matched[i]) continue;
                
                for (size_t j = 0; j < tracks_.size(); ++j) {
                    if (track_matched[j]) continue;
                    if (tracks_[j].state != TrackState::Tracked) continue;
                    
                    if (iou_matrix[i][j] > max_iou) {
                        max_iou = iou_matrix[i][j];
                        best_det = static_cast<int>(i);
                        best_track = static_cast<int>(j);
                    }
                }
            }
            
            if (max_iou < iou_threshold_) break;
            
            matches.emplace_back(best_det, best_track);
            det_matched[best_det] = true;
            track_matched[best_track] = true;
        }
        
        return matches;
    }
    
    float CalculateIOU(const cv::Rect& box1, const cv::Rect& box2) {
        cv::Rect intersection = box1 & box2;
        if (intersection.width <= 0 || intersection.height <= 0) {
            return 0.0f;
        }
        
        float intersection_area = static_cast<float>(intersection.area());
        float union_area = static_cast<float>(box1.area() + box2.area()) - intersection_area;
        
        return intersection_area / union_area;
    }
    
    std::vector<Track> tracks_;
    int next_id_ = 1;
    int frame_count_ = 0;
    int max_age_ = 30;
    float iou_threshold_ = 0.3f;
};

std::unique_ptr<ObjectTracker> CreateByteTracker() {
    return std::make_unique<ByteTracker>();
}

} // namespace face_recognition