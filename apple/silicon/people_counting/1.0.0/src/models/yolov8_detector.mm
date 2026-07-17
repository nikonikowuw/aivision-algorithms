/**
 * @file yolov8_detector.mm
 * @brief YOLOv8 model wrapper for CoreML implementation
 */

#include "yolov8_detector.h"
#include "postprocess/nms.h"
#include "common/logger.h"
#include <algorithm>

namespace people_count {

YOLOv8Detector::YOLOv8Detector(std::shared_ptr<CoreMLBackend> backend,
                               float conf_threshold,
                               float iou_threshold)
    : backend_(backend), conf_threshold_(conf_threshold), iou_threshold_(iou_threshold) {
    input_blob_.resize(1 * 3 * 640 * 640);
    letterbox_buffer_.resize(640 * 640 * 3);
}

YOLOv8Detector::~YOLOv8Detector() = default;

bool YOLOv8Detector::Load(const std::string& model_path) {
    if (!backend_) return false;
    return backend_->Load(model_path);
}

static void RestoreBBoxToOriginal(const image_utils::LetterboxInfo& info, int orig_w, int orig_h, Rect* bbox) {
    float x = (bbox->x - info.pad_x) / info.scale;
    float y = (bbox->y - info.pad_y) / info.scale;
    float w = bbox->width / info.scale;
    float h = bbox->height / info.scale;

    bbox->x = std::max(0.0f, std::min(x, static_cast<float>(orig_w)));
    bbox->y = std::max(0.0f, std::min(y, static_cast<float>(orig_h)));
    bbox->width = std::max(0.0f, std::min(w, static_cast<float>(orig_w - bbox->x)));
    bbox->height = std::max(0.0f, std::min(h, static_cast<float>(orig_h - bbox->y)));
}

bool YOLOv8Detector::Detect(const Image& frame_img, std::vector<DetectedObject>* results) {
    if (!results) return false;
    results->clear();

    // 1. Preprocess: Letterbox to 640x640
    image_utils::LetterboxInfo info;
    image_utils::Letterbox(frame_img, 640, letterbox_buffer_.data(), &info, &resize_buffer_);

    // 2. Preprocess: CHW flat float blob. YOLOv8 expects [0.0, 1.0] range (mean=0.0, std=255.0) and RGB order (swap_rgb=true)
    if (!image_utils::BlobFromImage(letterbox_buffer_.data(), 640, 640, input_blob_.data(), input_blob_.size(), true, 0.0f, 255.0f)) {
        ALGO_LOGE(DETECTOR, "Failed to create blob from image.");
        return false;
    }

    // 3. Inference
    int64_t shape[] = {1, 3, 640, 640};
    ModelInput model_in;
    model_in.data = input_blob_.data();
    model_in.size = input_blob_.size();
    model_in.shape = shape;
    model_in.shape_len = 4;

    std::vector<ModelInput> inputs = {model_in};
    if (!backend_->Run(inputs, &raw_outputs_)) {
        ALGO_LOGE(DETECTOR, "Inference execution failed.");
        return false;
    }

    if (raw_outputs_.empty()) {
        ALGO_LOGE(DETECTOR, "Inference returned empty outputs.");
        return false;
    }

    // 4. Postprocess: Decode YOLOv8 and apply NMS
    const auto& out = raw_outputs_[0];
    std::vector<DetectedObject> candidates = DecodeYOLOv8(out.buffer.data(), out.shape, conf_threshold_, iou_threshold_, 640, 640);

    // 5. Restore coordinates to original frame space
    for (auto& obj : candidates) {
        RestoreBBoxToOriginal(info, frame_img.width, frame_img.height, &obj.bbox);
        results->push_back(obj);
    }

    return true;
}

} // namespace people_count
