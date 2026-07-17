/**
 * @file yolo11_model.h
 * @brief GPU Face Recognition — YOLO11n person detection model wrapper.
 * @module Model Layer
 */

#ifndef GPU_FACE_RECOGNITION_YOLO11_MODEL_H
#define GPU_FACE_RECOGNITION_YOLO11_MODEL_H

#include "model_interface.h"
#include "preprocess/image_utils.h"
#include "postprocess/yolo_decoder.h"
#include <memory>

namespace face_rec {

/**
 * @class Yolo11Model
 * @brief YOLO11n person detection model.
 *        Pipeline: letterbox preprocess → TensorRT inference → YoloDecoder decode.
 */
class Yolo11Model : public IPersonDetector {
public:
    Yolo11Model();
    ~Yolo11Model() override;

    /**
     * @brief Initialize the model with a TensorRT backend and engine path.
     * @param backend Shared TensorRT backend (pre-loaded with engine)
     * @param input_size Network input size (default: 640)
     */
    bool Initialize(std::shared_ptr<IInferenceBackend> backend, int input_size = 640);

    std::vector<std::vector<DetectedObject>> DetectBatch(
        const std::vector<Image>& images,
        float conf_thres, float nms_iou_thres) override;

private:
    std::shared_ptr<IInferenceBackend> backend_;
    std::unique_ptr<YoloDecoder> decoder_;
    int input_size_ = 640;

    // Reusable buffers
    std::vector<uint8_t> letterbox_buffer_;
    std::vector<float> blob_buffer_;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_YOLO11_MODEL_H
