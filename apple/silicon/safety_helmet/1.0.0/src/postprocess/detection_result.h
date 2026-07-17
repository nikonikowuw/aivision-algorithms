#ifndef SAFETY_HELMET_DETECTION_RESULT_H
#define SAFETY_HELMET_DETECTION_RESULT_H

#include <string>

namespace safety_helmet {

struct Detection {
    float x, y, w, h;
    float confidence;
    int category_code;
    std::string category_name;
};

struct RawDetection {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_DETECTION_RESULT_H
