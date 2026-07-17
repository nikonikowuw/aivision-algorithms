/**
 * @file retinaface_model.cpp
 * @brief GPU Face Recognition — RetinaFace model implementation.
 */

#include "retinaface_model.h"
#include "common/logger.h"
#include "preprocess/image_utils.h"
#include <vector>

namespace face_rec {

RetinaFaceModel::RetinaFaceModel() : decoder_(std::make_unique<RetinaDecoder>()) {}
RetinaFaceModel::~RetinaFaceModel() = default;

bool RetinaFaceModel::Initialize(std::shared_ptr<IInferenceBackend> backend, int input_size) {
    backend_ = backend;
    input_size_ = input_size;
    ALGO_LOGI(FACE_DET, "RetinaFace model initialized (input=%d)", input_size_);
    return true;
}

std::vector<std::vector<DetectedObject>> RetinaFaceModel::DetectBatch(
    const std::vector<Image>& images,
    float conf_thres, float nms_iou_thres) {

    if (!backend_ || images.empty()) return {};

    size_t batch_size = images.size();
    std::vector<std::vector<float>> raw_outputs(batch_size);
    std::vector<int> orig_widths(batch_size), orig_heights(batch_size);

    // Preprocess: resize to input_size and normalize
    std::vector<std::vector<float>> batch_data(batch_size);
    std::vector<ModelInput> inputs(batch_size);
    std::vector<uint8_t> resize_buffer(input_size_ * input_size_ * 3);

    for (size_t i = 0; i < batch_size; ++i) {
        orig_widths[i] = images[i].width;
        orig_heights[i] = images[i].height;

        // Resize to network input size
        BilinearResize(images[i].data, images[i].width, images[i].height, images[i].width,
                       resize_buffer.data(), input_size_, input_size_);

        // Normalize: (pixel - 127.5) / 127.5
        batch_data[i].resize(3 * input_size_ * input_size_);
        BgrToPlanarFloat(resize_buffer.data(), batch_data[i].data(),
                         input_size_, input_size_);

        inputs[i].data = batch_data[i].data();
        inputs[i].size = batch_data[i].size();
        inputs[i].shape = reinterpret_cast<const int64_t*>(&input_size_);
        inputs[i].shape_len = 4;
    }

    std::vector<ModelOutput> outputs;
    if (!backend_->Run(inputs, &outputs)) {
        ALGO_LOGE(FACE_DET, "RetinaFace inference failed");
        return std::vector<std::vector<DetectedObject>>(batch_size);
    }

    if (!outputs.empty()) {
        for (size_t i = 0; i < batch_size; ++i) {
            raw_outputs[i] = outputs[0].buffer;
        }
    }

    return decoder_->DecodeBatch(raw_outputs, orig_widths, orig_heights,
                                  input_size_, input_size_,
                                  conf_thres, nms_iou_thres);
}

} // namespace face_rec
