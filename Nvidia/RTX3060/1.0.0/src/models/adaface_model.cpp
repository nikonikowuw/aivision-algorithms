/**
 * @file adaface_model.cpp
 * @brief GPU Face Recognition — AdaFace model implementation.
 */

#include "adaface_model.h"
#include "common/logger.h"
#include "preprocess/image_utils.h"
#include <vector>
#include <cmath>

namespace face_rec {

AdaFaceModel::AdaFaceModel() = default;
AdaFaceModel::~AdaFaceModel() = default;

bool AdaFaceModel::Initialize(std::shared_ptr<IInferenceBackend> backend) {
    backend_ = backend;
    ALGO_LOGI(FACE_REC, "AdaFace IR-50 model initialized");
    return true;
}

void AdaFaceModel::L2Normalize(std::vector<float>& embedding) {
    float norm = 0.0f;
    for (float v : embedding) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : embedding) {
            v /= norm;
        }
    }
}

std::vector<std::vector<float>> AdaFaceModel::ExtractBatch(
    const std::vector<std::vector<uint8_t>>& aligned_faces) {

    if (!backend_ || aligned_faces.empty()) return {};

    constexpr int kInputSize = 112;
    constexpr int kEmbeddingDim = 512;
    size_t batch_size = aligned_faces.size();

    // Preprocess: each aligned face is 112×112 BGR → CHW float RGB
    std::vector<std::vector<float>> batch_data(batch_size);
    std::vector<ModelInput> inputs(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        batch_data[i].resize(3 * kInputSize * kInputSize);

        // Create Image view over the aligned face buffer
        Image face_view;
        face_view.data = aligned_faces[i].data();
        face_view.width = kInputSize;
        face_view.height = kInputSize;
        face_view.channels = 3;
        face_view.stride = kInputSize;

        BgrToPlanarFloat(face_view.data, batch_data[i].data(),
                         kInputSize, kInputSize);

        inputs[i].data = batch_data[i].data();
        inputs[i].size = batch_data[i].size();
        inputs[i].shape = reinterpret_cast<const int64_t*>(&kInputSize);
        inputs[i].shape_len = 4;
    }

    std::vector<ModelOutput> outputs;
    if (!backend_->Run(inputs, &outputs)) {
        ALGO_LOGE(FACE_REC, "AdaFace inference failed");
        return {};
    }

    // Parse outputs: expect [batch, 512] embedding
    std::vector<std::vector<float>> embeddings(batch_size);

    if (!outputs.empty()) {
        const auto& emb_buf = outputs[0].buffer;
        for (size_t i = 0; i < batch_size; ++i) {
            size_t offset = i * kEmbeddingDim;
            if (offset + kEmbeddingDim <= emb_buf.size()) {
                embeddings[i].assign(emb_buf.begin() + offset,
                                     emb_buf.begin() + offset + kEmbeddingDim);
                L2Normalize(embeddings[i]);
            }
        }
    }

    return embeddings;
}

} // namespace face_rec
