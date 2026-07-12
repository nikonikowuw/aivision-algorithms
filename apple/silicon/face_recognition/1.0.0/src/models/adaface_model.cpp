/**
 * @file adaface_model.cpp
 * @brief AdaFace 模型实现：BGR 归一化到 [-1,1] 和 L2 归一化 512-d embedding
 *        AdaFace model implementation: BGR normalization to [-1,1] and L2-normalized 512-d embedding
 */
#include "adaface_model.h"
#include "common/logger.h"
#include <cmath>

namespace face_rec {

AdaFaceModel::AdaFaceModel(float recognition_threshold) {
    (void)recognition_threshold;
    
    // 预分配 112×112×3 float 缓冲区 (NCHW 格式)
    // Pre-allocate 112×112×3 float buffer (NCHW format)
    input_buffer_.resize(1 * 3 * 112 * 112);
    input_shape_ = {1, 3, 112, 112};
    output_names_ = {"output"};
}

bool AdaFaceModel::Load(const std::string& model_path,
                       std::shared_ptr<IInferenceBackend> backend) {
    backend_ = backend;
    if (!backend_) return false;
    return backend_->Load(model_path);
}

bool AdaFaceModel::Preprocess(const Image& aligned_face, ModelInput* input,
                              image_utils::LetterboxInfo* info) {
    (void)info;
    // 检查对齐人脸尺寸是否严格为 112×112
    // Verify aligned face size is strictly 112×112
    if (!aligned_face.data || aligned_face.width != 112 || aligned_face.height != 112) {
        ALGO_LOGE(BACKEND, "AdaFace preprocess failed: Invalid aligned face dimensions.");
        return false;
    }
    
    // Normalization to [-1.0, 1.0]: (x - 127.5) / 127.5, maintaining BGR CHW layout
    // 归一化到 [-1.0, 1.0]: (x - 127.5) / 127.5，保持 BGR CHW 排布
    // bgr=true 表示保持 BGR 通道顺序（AdaFace 使用 BGR 而非 RGB）
    // bgr=true means keep BGR channel order (AdaFace uses BGR instead of RGB)
    if (!image_utils::BlobFromImage(aligned_face.data, 112, 112, input_buffer_.data(), input_buffer_.size(), false, 127.5f, 127.5f)) {
        return false;
    }
    
    input->data = input_buffer_.data();
    input->size = input_buffer_.size();
    input->shape = input_shape_.data();
    input->shape_len = input_shape_.size();
    
    return true;
}

bool AdaFaceModel::Postprocess(const std::vector<ModelOutput>& raw_outputs,
                               const image_utils::LetterboxInfo& info,
                               std::vector<DetectedObject>* results) {
    (void)info;
    // 检查推理输出是否有效
    // Check if inference output is valid
    if (raw_outputs.empty() || raw_outputs[0].buffer.empty()) {
        ALGO_LOGE(BACKEND, "AdaFace postprocess failed: No outputs received from backend.");
        return false;
    }
    
    const float* raw_emb = raw_outputs[0].buffer.data();
    size_t dim = raw_outputs[0].buffer.size();
    
    if (dim != 512) {
        ALGO_LOGE(BACKEND, "AdaFace postprocess failed: Expected 512-d embedding, got %d", static_cast<int>(dim));
        return false;
    }
    
    // Calculate L2 norm of the embedding vector
    // 计算嵌入向量的 L2 范数
    float l2_norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        l2_norm += raw_emb[i] * raw_emb[i];
    }
    l2_norm = std::sqrt(l2_norm);
    if (l2_norm < 1e-6f) l2_norm = 1e-6f;  // 防止除零 / prevent division by zero
    
    // Fill result vector with normalized embedding
    // 将归一化的嵌入向量填入结果结构体
    if (results && !results->empty()) {
        auto& obj = results->front();
        obj.embedding.resize(dim);
        for (size_t i = 0; i < dim; ++i) {
            obj.embedding[i] = raw_emb[i] / l2_norm;
        }
    }
    
    return true;
}

void AdaFaceModel::Unload() {
    if (backend_) {
        backend_->Unload();
    }
}

} // namespace face_rec
