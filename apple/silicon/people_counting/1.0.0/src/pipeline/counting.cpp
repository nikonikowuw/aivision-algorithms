/**
 * @file counting.cpp
 * @brief Line crossing detection implementation
 */

#include "counting.h"
#include "common/logger.h"
#include <algorithm>

namespace people_count {

LineCounter::LineCounter() = default;
LineCounter::~LineCounter() = default;

void LineCounter::Reset() {
    last_points_.clear();
    crossed_lines_.clear();
}

static inline float CrossProduct(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

static inline bool IsIntersect(Point p1, Point p2, Point q1, Point q2) {
    float cp1 = CrossProduct(p1, p2, q1);
    float cp2 = CrossProduct(p1, p2, q2);
    float cp3 = CrossProduct(q1, q2, p1);
    float cp4 = CrossProduct(q1, q2, p2);

    return (((cp1 > 0.0f && cp2 < 0.0f) || (cp1 < 0.0f && cp2 > 0.0f)) &&
            ((cp3 > 0.0f && cp4 < 0.0f) || (cp3 < 0.0f && cp4 > 0.0f)));
}

void LineCounter::Update(const std::vector<CountingLine>& lines,
                         std::vector<DetectedObject>* objects,
                         int* count_in, int* count_out) {
    if (!objects || !count_in || !count_out) return;

    std::unordered_set<int> current_track_ids;

    for (auto& obj : *objects) {
        if (obj.track_id < 0) continue;
        current_track_ids.insert(obj.track_id);

        // Footpoint is bottom center of the bounding box
        Point curr_pt;
        curr_pt.x = obj.bbox.x + obj.bbox.width / 2.0f;
        curr_pt.y = obj.bbox.y + obj.bbox.height;

        auto it = last_points_.find(obj.track_id);
        if (it != last_points_.end()) {
            Point prev_pt = it->second;

            // Check intersection with each config line
            for (const auto& line : lines) {
                if (crossed_lines_[obj.track_id].count(line.id) > 0) {
                    continue; // Already counted for this line
                }

                if (IsIntersect(prev_pt, curr_pt, line.start, line.end)) {
                    // Crossed! Determine direction
                    std::string direction = "none";
                    bool is_in = false;

                    if (line.in_direction == "down") {
                        is_in = (curr_pt.y > prev_pt.y);
                    } else if (line.in_direction == "up") {
                        is_in = (curr_pt.y < prev_pt.y);
                    } else if (line.in_direction == "right") {
                        is_in = (curr_pt.x > prev_pt.x);
                    } else if (line.in_direction == "left") {
                        is_in = (curr_pt.x < prev_pt.x);
                    } else {
                        // Default logic if unknown direction: check Y movement
                        is_in = (curr_pt.y > prev_pt.y);
                    }

                    if (is_in) {
                        direction = "in";
                        (*count_in)++;
                        ALGO_LOGI(PIPELINE, "Track %d crossed line %s IN. Total IN: %d", obj.track_id, line.id.c_str(), *count_in);
                    } else {
                        direction = "out";
                        (*count_out)++;
                        ALGO_LOGI(PIPELINE, "Track %d crossed line %s OUT. Total OUT: %d", obj.track_id, line.id.c_str(), *count_out);
                    }

                    obj.crossed = true;
                    obj.direction = direction;
                    crossed_lines_[obj.track_id].insert(line.id);
                }
            }
        }

        // Update the last seen position of the track
        last_points_[obj.track_id] = curr_pt;
    }

    // Clean up tracks that are no longer present to prevent memory leaks
    for (auto it = last_points_.begin(); it != last_points_.end();) {
        if (current_track_ids.count(it->first) == 0) {
            crossed_lines_.erase(it->first);
            it = last_points_.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace people_count
