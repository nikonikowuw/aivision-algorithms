/**
 * @file yolo11_model.cpp
 * @brief GPU Face Recognition — YOLO11n model implementation.
 */

#include "yolo11_model.h"
#include "common/logger.h"
#include <vector>
#include <algorithm>

namespace face_rec {

Yolo11Model::Yolo11Model() : decoder_(std::make_unique<YoloDecoder>()) {}
Yolo11Model::~Yolo11Model() = default;

bool Yolo11Model::Initialize(std::shared_ptr<IInferenceBackend> backend, int input_size) {
    backend_ = backend;
    input_size_ = input_size;
    letterbox_buffer_.resize(input_size * input_size * 3);
    blob_buffer_.resize(3 * input_size * input_size);
    ALGO_LOGI(PERSON_DET, "YOLO11n model initialized (input=%d)", input_size_);
    return true;
}

std::vector<std::vector<DetectedObject>> Yolo11Model::DetectBatch(
    const std::vector<Image>& images,
    float conf_thres, float nms_iou_thres) {

    if (!backend_ || images.empty()) return {};

    size_t batch_size = images.size();
    std::vector<std::vector<float>> raw_outputs(batch_size);
    std::vector<int> orig_widths(batch_size), orig_heights(batch_size);
    std::vector<float> scales(batch_size);
    std::vector<int> pad_xs(batch_size), pad_ys(batch_size);

    // Preprocess each image: letterbox + normalize
    std::vector<ModelInput> inputs;
    inputs.resize(batch_size);

    std::vector<std::vector<float>> batch_data(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        orig_widths[i] = images[i].width;
        orig_heights[i] = images[i].height;

        // Letterbox resize
        LetterboxResult lr = LetterboxResize(
            images[i].data, images[i].width, images[i].height, images[i].width,
            letterbox_buffer_.data(), input_size_, input_size_);
        scales[i] = lr.scale;
        pad_xs[i] = lr.pad_x;
        pad_ys[i] = lr.pad_y;

        // Convert to planar float (CHW, RGB, normalized)
        batch_data[i].resize(3 * input_size_ * input_size_);
        BgrToPlanarFloat(letterbox_buffer_.data(), batch_data[i].data(),
                         input_size_, input_size_);

        inputs[i].data = batch_data[i].data();
        inputs[i].size = batch_data[i].size();
        inputs[i].shape = reinterpret_cast<const int64_t*>(&input_size_);
        inputs[i].shape_len = 4;  // [batch, C, H, W] — batch dim added by TRT
    }

    // Run TensorRT inference
    std::vector<ModelOutput> outputs;
    if (!backend_->Run(inputs, &outputs)) {
        ALGO_LOGE(PERSON_DET, "YOLO11n inference failed");
        return std::vector<std::vector<DetectedObject>>(batch_size);
    }

    // Decode outputs
    // Note: For simplicity in this skeleton, we process single-image outputs
    // The batch inference outputs are per-binding, not per-image in TRT
    // In a full implementation, the output shape handles batch dim
    if (!outputs.empty()) {
        for (size_t i = 0; i < batch_size; ++i) {
            raw_outputs[i] = outputs[0].buffer;  // Simplified: same output for batch
        }
    }

    return decoder_->DecodeBatch(raw_outputs, orig_widths, orig_heights,
                                  scales, pad_xs, pad_ys,
                                  conf_thres, nms_iou_thres);
}

} // namespace face_rec
