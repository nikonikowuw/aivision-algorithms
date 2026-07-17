#ifndef SAFETY_HELMET_PIPELINE_H
#define SAFETY_HELMET_PIPELINE_H

#include "common/config_parser.h"
#include "models/damoyolo_model.h"
#include "postprocess/detection_result.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>

namespace safety_helmet {

class SafetyHelmetPipeline {
public:
    SafetyHelmetPipeline() = default;
    ~SafetyHelmetPipeline() = default;

    bool Initialize(const char* config_json);
    bool Detect(const cv::Mat& frame, std::vector<Detection>& results);
    void Destroy();

    const AlgoConfig& GetConfig() const { return config_; }

private:
    AlgoConfig config_;
    DamoYoloModel model_;
    bool initialized_ = false;
    std::mutex mutex_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_PIPELINE_H
