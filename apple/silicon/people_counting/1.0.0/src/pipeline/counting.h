/**
 * @file counting.h
 * @brief Line crossing detection and counting
 */

#ifndef PEOPLE_COUNTING_COUNTING_H
#define PEOPLE_COUNTING_COUNTING_H

#include "common/types.h"
#include "common/config.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace people_count {

class LineCounter {
public:
    LineCounter();
    ~LineCounter();

    void Update(const std::vector<CountingLine>& lines,
                std::vector<DetectedObject>* objects,
                int* count_in, int* count_out);

    void Reset();

private:
    // Maps track_id to its last seen footpoint (bottom center of bbox)
    std::unordered_map<int, Point> last_points_;

    // Maps track_id to set of line IDs it has already crossed
    std::unordered_map<int, std::unordered_set<std::string>> crossed_lines_;
};

} // namespace people_count

#endif // PEOPLE_COUNTING_COUNTING_H
