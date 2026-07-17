#ifndef SAFETY_HELMET_LETTERBOX_H
#define SAFETY_HELMET_LETTERBOX_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace safety_helmet {

struct LetterBoxInfo {
    float ratio = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int orig_w = 0;
    int orig_h = 0;
};

bool LetterBoxPreprocess(const cv::Mat& src, cv::Mat& dst, std::vector<float>& input_tensor, int target_size, LetterBoxInfo& info);

} // namespace safety_helmet

#endif // SAFETY_HELMET_LETTERBOX_H
