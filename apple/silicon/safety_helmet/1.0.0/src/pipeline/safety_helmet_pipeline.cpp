#include "safety_helmet_pipeline.h"
#include "preprocess/letterbox.h"
#include "postprocess/damoyolo_decoder.h"
#include "postprocess/nms.h"
#include "postprocess/coord_mapper.h"
#include "common/logger.h"

namespace safety_helmet {

bool SafetyHelmetPipeline::Initialize(const char* config_json) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;

    try {
        config_ = AlgoConfig::LoadConfig(config_json);
        
        ALGO_LOG_INFO("Initializing pipeline. Model path: %s, Conf: %.2f, IoU: %.2f",
                      config_.model_path.c_str(), config_.conf_threshold, config_.iou_threshold);

        if (!model_.Init(config_.model_path)) {
            ALGO_LOG_ERROR("Failed to initialize DAMO-YOLO model");
            return false;
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("Pipeline initialization failed with exception: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("Pipeline initialization failed with unknown exception");
        return false;
    }
}

bool SafetyHelmetPipeline::Detect(const cv::Mat& frame, std::vector<Detection>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();

    if (!initialized_) {
        ALGO_LOG_ERROR("Pipeline not initialized");
        return false;
    }

    if (frame.empty()) {
        ALGO_LOG_ERROR("Frame is empty");
        return false;
    }

    try {
        // 1. Preprocess (Letterbox + float conversion)
        cv::Mat processed_img;
        std::vector<float> input_tensor;
        LetterBoxInfo box_info;
        if (!LetterBoxPreprocess(frame, processed_img, input_tensor, 640, box_info)) {
            ALGO_LOG_ERROR("Preprocessing failed");
            return false;
        }

        // 2. Inference
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        std::vector<std::vector<float>> output_tensors;
        std::vector<std::vector<int64_t>> output_shapes;
        if (!model_.Forward(input_tensor, input_shape, output_tensors, output_shapes)) {
            ALGO_LOG_ERROR("Model inference forward failed");
            return false;
        }

        // 3. Output decode
        std::vector<RawDetection> raw_detections;
        if (!DecodeOutputs(output_tensors, output_shapes, config_.conf_threshold, raw_detections)) {
            ALGO_LOG_ERROR("Decoding model outputs failed");
            return false;
        }

        // 4. Non-Maximum Suppression (NMS)
        std::vector<RawDetection> filtered_raw = ApplyNMS(raw_detections, config_.iou_threshold);

        // 5. Coordinate mapping and normalization
        for (const auto& raw_det : filtered_raw) {
            results.push_back(MapToOriginal(raw_det, box_info));
        }

        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("Detect exception: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("Detect unknown exception");
        return false;
    }
}

void SafetyHelmetPipeline::Destroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    ALGO_LOG_INFO("Pipeline destroyed");
}

} // namespace safety_helmet
