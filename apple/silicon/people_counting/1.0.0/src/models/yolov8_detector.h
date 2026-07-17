/**
 * @file yolov8_detector.h
 * @brief YOLOv8 model wrapper for CoreML
 */

#ifndef PEOPLE_COUNTING_YOLOV8_DETECTOR_H
#define PEOPLE_COUNTING_YOLOV8_DETECTOR_H

#include "common/types.h"
#include "common/config.h"
#include "runtime/coreml_backend.h"
#include "preprocess/image_utils.h"
#include <memory>
#include <vector>

namespace people_count {

class YOLOv8Detector {
public:
    explicit YOLOv8Detector(std::shared_ptr<CoreMLBackend> backend,
                            float conf_threshold = 0.5f,
                            float iou_threshold = 0.45f);
    ~YOLOv8Detector();

    bool Load(const std::string& model_path);
    bool Detect(const Image& frame_img, std::vector<DetectedObject>* results);

private:
    std::shared_ptr<CoreMLBackend> backend_;
    float conf_threshold_;
    float iou_threshold_;

    std::vector<float> input_blob_;
    std::vector<uint8_t> letterbox_buffer_;
    std::vector<uint8_t> resize_buffer_;
    std::vector<ModelOutput> raw_outputs_;
};

} // namespace people_count

#endif // PEOPLE_COUNTING_YOLOV8_DETECTOR_H
