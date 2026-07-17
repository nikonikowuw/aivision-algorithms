#include "letterbox.h"

namespace safety_helmet {

bool LetterBoxPreprocess(const cv::Mat& src, cv::Mat& dst, std::vector<float>& input_tensor, int target_size, LetterBoxInfo& info) {
    if (src.empty()) {
        return false;
    }

    info.orig_w = src.cols;
    info.orig_h = src.rows;

    float r = std::min(static_cast<float>(target_size) / src.cols, static_cast<float>(target_size) / src.rows);
    info.ratio = r;

    int new_unpad_w = static_cast<int>(std::round(src.cols * r));
    int new_unpad_h = static_cast<int>(std::round(src.rows * r));

    cv::Mat tmp;
    if (src.cols != new_unpad_w || src.rows != new_unpad_h) {
        cv::resize(src, tmp, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        tmp = src.clone();
    }

    float dw = static_cast<float>(target_size - new_unpad_w);
    float dh = static_cast<float>(target_size - new_unpad_h);

    dw /= 2.0f;
    dh /= 2.0f;

    info.pad_x = dw;
    info.pad_y = dh;

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(dst, rgb, cv::COLOR_BGR2RGB);

    // Split channels and convert
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    input_tensor.resize(3 * target_size * target_size);
    float* ptr = input_tensor.data();

    for (int c = 0; c < 3; ++c) {
        cv::Mat dest_channel(target_size, target_size, CV_32FC1, ptr + c * target_size * target_size);
        channels[c].convertTo(dest_channel, CV_32FC1, 1.0 / 255.0);
    }

    return true;
}

} // namespace safety_helmet
