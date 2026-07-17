/**
 * @file pipeline.cpp
 * @brief PipelineOrchestrator implementation
 */

#include "pipeline.h"
#include "common/logger.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace people_count {

PipelineOrchestrator::~PipelineOrchestrator() {
    Destroy();
}

bool PipelineOrchestrator::Initialize(const char* config_json) {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    if (initialized_) return true;

    try {
        // 1. Load config
        config_ = Config::LoadConfig(config_json);

        // 2. Initialize inference backend
        backend_ = std::make_shared<CoreMLBackend>();
        
        // 3. Initialize detector
        detector_ = std::make_unique<YOLOv8Detector>(backend_, config_.conf_threshold, config_.iou_threshold);
        std::string absolute_model_path = JoinPath(config_.package_dir, config_.model_path);
        if (!detector_->Load(absolute_model_path)) {
            ALGO_LOGE(PIPELINE, "Failed to load YOLOv8 model from: %s", absolute_model_path.c_str());
            return false;
        }

        // 4. Initialize tracker
        if (config_.enable_tracker) {
            tracker_ = std::make_unique<ByteTracker>(config_.match_threshold, config_.track_buffer);
        }

        // 5. Initialize line counter
        counter_ = std::make_unique<LineCounter>();

        timing_.Reset();
        frame_count_ = 0;
        count_in_ = 0;
        count_out_ = 0;
        initialized_ = true;

        ALGO_LOGI(PIPELINE, "Pipeline Orchestrator initialized successfully.");
        return true;
    } catch (const std::exception& e) {
        ALGO_LOGE(PIPELINE, "Exception during pipeline initialization: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOGE(PIPELINE, "Unknown exception during pipeline initialization.");
        return false;
    }
}

int PipelineOrchestrator::Infer(const hw_buffer_desc_t* input, const char* context_json,
                                 infer_result_t* result) {
    if (!input || !result) return -1;
    result->result_json = nullptr;
    result->result_json_len = 0;
    result->infer_time_us = 0;

    std::lock_guard<std::mutex> lock(infer_mutex_);
    if (!initialized_) return -1;

    timing_.Reset();
    timing_.total.Start();

    int frame_w = static_cast<int>(input->width);
    int frame_h = static_cast<int>(input->height);

    if (frame_w <= 0 || frame_h <= 0 || !input->data) {
        ALGO_LOGE(PIPELINE, "Infer failed: Invalid input frame parameters.");
        return -2;
    }

    // ---- Step 1: Decode Frame ----
    timing_.preprocess.Start();
    Image frame_img;
    if (input->pixel_format == pixel_format::BGR24) {
        if (input->stride < input->width * 3 ||
            input->size < static_cast<size_t>(input->stride) * input->height) {
            ALGO_LOGE(PIPELINE, "Infer failed: Invalid BGR24 stride or buffer size.");
            return -2;
        }
        frame_img = Image{reinterpret_cast<const uint8_t*>(input->data), frame_w, frame_h, 3, static_cast<int>(input->stride)};
    } else if (input->pixel_format == pixel_format::NV12) {
        if ((frame_w & 1) != 0 || (frame_h & 1) != 0 || input->stride < input->width ||
            input->size < static_cast<size_t>(input->stride) * input->height * 3 / 2) {
            ALGO_LOGE(PIPELINE, "Infer failed: Invalid NV12 dimensions, stride, or buffer size.");
            return -2;
        }
        decoded_bgr_buffer_.resize(frame_w * frame_h * 3);
        if (!image_utils::NV12ToBGR(reinterpret_cast<const uint8_t*>(input->data),
                                    frame_w, frame_h, static_cast<int>(input->stride),
                                    static_cast<int>(input->stride), decoded_bgr_buffer_.data())) {
            ALGO_LOGE(PIPELINE, "Infer failed: NV12 conversion failed.");
            return -2;
        }
        frame_img = Image{decoded_bgr_buffer_.data(), frame_w, frame_h, 3, frame_w * 3};
    } else {
        ALGO_LOGE(PIPELINE, "Infer failed: Unsupported pixel format: 0x%08x", input->pixel_format);
        return -2;
    }
    timing_.preprocess.Stop();

    // ---- Step 2: YOLOv8 Person Detection ----
    timing_.body_detect.Start();
    std::vector<DetectedObject> detections;
    if (!detector_->Detect(frame_img, &detections)) {
        ALGO_LOGE(PIPELINE, "YOLOv8 detection failed.");
        return -3;
    }
    timing_.body_detect.Stop();

    // ---- Step 3: BBox Coordinate Normalization ----
    // Bbox coordinates must be normalized relative to original frame width and height
    for (auto& obj : detections) {
        obj.bbox.x /= frame_w;
        obj.bbox.y /= frame_h;
        obj.bbox.width /= frame_w;
        obj.bbox.height /= frame_h;
    }

    // ---- Step 4: Tracking ----
    timing_.tracker.Start();
    std::vector<DetectedObject> tracked_objects;
    if (config_.enable_tracker && tracker_) {
        tracked_objects = tracker_->Update(detections);
    } else {
        tracked_objects = detections;
        for (auto& obj : tracked_objects) {
            obj.track_id = -1;
        }
    }
    timing_.tracker.Stop();

    // ---- Step 5: Line Crossing Detection ----
    timing_.counting.Start();
    if (context_json) {
        std::vector<CountingLine> lines = ParseCountingLines(context_json);
        if (!lines.empty() && counter_) {
            counter_->Update(lines, &tracked_objects, &count_in_, &count_out_);
        }
    }
    timing_.counting.Stop();

    // ---- Step 6: Construct JSON Output ----
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"detections\": [\n";
    for (size_t i = 0; i < tracked_objects.size(); ++i) {
        const auto& obj = tracked_objects[i];
        ss << "    {\n";
        ss << "      \"category_code\": " << obj.label << ",\n";
        ss << "      \"detect_confidence\": " << std::fixed << std::setprecision(4) << obj.confidence << ",\n";
        ss << "      \"bbox\": [" 
           << std::fixed << std::setprecision(6) 
           << obj.bbox.x << ", " 
           << obj.bbox.y << ", " 
           << obj.bbox.width << ", " 
           << obj.bbox.height << "],\n";
        ss << "      \"track_id\": " << obj.track_id << ",\n";
        ss << "      \"crossed\": " << (obj.crossed ? "true" : "false") << ",\n";
        ss << "      \"direction\": ";
        if (obj.direction == "none") {
            ss << "null\n";
        } else {
            ss << "\"" << obj.direction << "\"\n";
        }
        ss << "    }" << (i + 1 < tracked_objects.size() ? "," : "") << "\n";
    }
    ss << "  ],\n";
    ss << "  \"summary\": {\n";
    ss << "    \"count_in\": " << count_in_ << ",\n";
    ss << "    \"count_out\": " << count_out_ << ",\n";
    ss << "    \"current_count\": " << tracked_objects.size() << "\n";
    ss << "  }\n";
    ss << "}";

    std::string json_str = ss.str();
    result->result_json_len = json_str.size();
    result->result_json = reinterpret_cast<char*>(std::malloc(result->result_json_len + 1));
    std::memcpy(result->result_json, json_str.c_str(), result->result_json_len + 1);

    timing_.total.Stop();
    result->infer_time_us = static_cast<uint32_t>(timing_.total.ElapsedMs() * 1000.0);

    frame_count_++;
    if (config_.enable_timing_report) {
        timing_.Report(config_.log_timing_interval, frame_count_);
    }

    return 0;
}

void PipelineOrchestrator::Destroy() {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    if (!initialized_) return;

    if (backend_) {
        backend_->Unload();
        backend_.reset();
    }
    detector_.reset();
    tracker_.reset();
    counter_.reset();
    initialized_ = false;
    ALGO_LOGI(PIPELINE, "Pipeline Orchestrator destroyed.");
}

} // namespace people_count
