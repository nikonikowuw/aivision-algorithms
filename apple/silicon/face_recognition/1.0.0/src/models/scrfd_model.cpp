/**
 * @file scrfd_model.cpp
 * @brief SCRFD 模型实现：预处理（letterbox+归一化+CHW）和后处理（stride 映射 → ScrfdDecoder）
 *        SCRFD model implementation: preprocessing (letterbox+normalization+CHW)
 *        and postprocessing (stride mapping → ScrfdDecoder)
 */
#include "scrfd_model.h"
#include "postprocess/scrfd_decoder.h"
#include "common/logger.h"

namespace face_rec {

ScrfdModel::ScrfdModel(int input_size, float conf_threshold, float iou_threshold, int max_count, bool is_person_model)
    : input_size_(input_size), conf_threshold_(conf_threshold), iou_threshold_(iou_threshold), max_count_(max_count) {

    // Allocate buffer for float input tensor: shape is [1, 3, input_size, input_size]
    // 预分配 float 输入缓冲区：shape 为 [1, 3, input_size, input_size]
    input_buffer_.resize(1 * 3 * input_size_ * input_size_);
    letterbox_buffer_.resize(static_cast<size_t>(input_size_) * input_size_ * 3);
    input_shape_ = {1, 3, input_size_, input_size_};

    // 设置归一化参数：行人模型用 (val-127.5)/127.5 → [-1,1]
    // 人脸模型用 原始 [0,255] 像素（mean=0, std=1）
    // Set normalization: person model uses (val-127.5)/127.5 → [-1,1]
    // Face model uses raw [0,255] pixels (mean=0, std=1)
    if (!is_person_model) {
        mean_ = 0.0f;
        std_ = 1.0f;
    }

    if (is_person_model) {
        // ---- 行人检测配置 / Person detector config ----
        // 5-stride 布局 (8, 16, 32, 64, 128)，每个 stride 1 个 anchor
        // 5-stride layout (8, 16, 32, 64, 128), 1 anchor per stride
        strides_ = {8, 16, 32, 64, 128};
        anchor_counts_ = {1, 1, 1, 1, 1};
        has_kps_ = true;
        
        // 输出名称按 stride 分组：前 5 个为 score，中间 5 个为 bbox，最后 5 个为 keypoints
        // Output names grouped by stride: first 5 = scores, middle 5 = bboxes, last 5 = keypoints
        // NOTE: CoreML output names (converted from ONNX)
        output_names_ = {
            "var_782", "var_883", "var_984", "var_1085", "var_1186", // Scores
            "var_796", "var_897", "var_998", "var_1099", "var_1200", // BBoxes
            "var_810", "var_911", "var_1012", "var_1113", "var_1214" // Keypoints
        };
    } else {
        // ---- 人脸检测配置 / Face detector config (is_person_model = false) ----
        // 3-stride 布局 (8, 16, 32)，每个 stride 2 个 anchor
        // 3-stride layout (8, 16, 32), 2 anchors per stride
        strides_ = {8, 16, 32};
        anchor_counts_ = {2, 2, 2};
        has_kps_ = true;

        // 输出名称：3 score + 3 bbox + 3 keypoints
        // NOTE: CoreML output names
        output_names_ = {
            "var_732", "var_839", "var_946", // Scores
            "var_748", "var_855", "var_962", // BBoxes
            "var_764", "var_871", "var_978"  // Keypoints
        };
    }
}

bool ScrfdModel::Load(const std::string& model_path,
                      std::shared_ptr<IInferenceBackend> backend) {
    backend_ = backend;
    if (!backend_) return false;
    return backend_->Load(model_path);
}

bool ScrfdModel::Preprocess(const Image& image, ModelInput* input,
                            image_utils::LetterboxInfo* info) {
    if (!image.data || image.width <= 0 || image.height <= 0) return false;
    
    // Step 1: Letterbox — 保持宽高比缩放到 input_size_，不足部分填充
    // Step 1: Letterbox — scale to input_size_ maintaining aspect ratio, pad remaining area
    image_utils::Letterbox(image, input_size_, letterbox_buffer_.data(), info, &resize_buffer_);
    
    // Step 2: BlobFromImage — BGR→RGB + 除以 mean 归一化 + HWC→CHW
    // Step 2: BlobFromImage — BGR→RGB + divide by mean normalization + HWC→CHW
    if (!image_utils::BlobFromImage(letterbox_buffer_.data(), input_size_, input_size_, input_buffer_.data(), input_buffer_.size(), true, mean_, std_)) {
        return false;
    }
    
    // 设置模型输入指针（零拷贝：直接指向预分配缓冲区）
    // Set model input pointers (zero-copy: point directly to pre-allocated buffer)
    input->data = input_buffer_.data();
    input->size = input_buffer_.size();
    input->shape = input_shape_.data();
    input->shape_len = input_shape_.size();
    
    return true;
}

bool ScrfdModel::Postprocess(const std::vector<ModelOutput>& raw_outputs,
                             const image_utils::LetterboxInfo& info,
                             std::vector<DetectedObject>* results) {
    (void)info;
    if (raw_outputs.size() != output_names_.size()) {
        ALGO_LOGE(BACKEND, "ScrfdModel postprocess failed: Expected %d outputs, got %d",
                  static_cast<int>(output_names_.size()), static_cast<int>(raw_outputs.size()));
        return false;
    }
    
    size_t num_strides = strides_.size();
    std::vector<ScrfdDecoder::StrideOutput> stride_outs(num_strides);
    
    // 将原始模型输出按 stride 组织成解码器所需的结构
    // Organize raw model outputs into StrideOutput structures required by the decoder
    for (size_t i = 0; i < num_strides; ++i) {
        const ModelOutput* score_out = nullptr;
        const ModelOutput* bbox_out = nullptr;
        const ModelOutput* kp_out = nullptr;
        
        // 按名称匹配：第 i 个 score 和第 i 个 bbox
        // Match by name: i-th score and i-th bbox
        std::string score_name = output_names_[i];
        std::string bbox_name = output_names_[i + num_strides];
        
        for (const auto& out : raw_outputs) {
            if (out.name == score_name) score_out = &out;
            else if (out.name == bbox_name) bbox_out = &out;
        }
        
        // 关键点输出在命名列表的后 1/3 段
        // Keypoint outputs are in the last 1/3 of the naming list
        if (has_kps_) {
            std::string kp_name = output_names_[i + 2 * num_strides];
            for (const auto& out : raw_outputs) {
                if (out.name == kp_name) kp_out = &out;
            }
        }
        
        // Fallback to sequential index if not found by name
        // 如果按名称未找到，则按顺序索引回退
        if (!score_out) score_out = &raw_outputs[i];
        if (!bbox_out) bbox_out = &raw_outputs[i + num_strides];
        if (has_kps_ && !kp_out) kp_out = &raw_outputs[i + 2 * num_strides];
        
        stride_outs[i].scores = score_out->buffer.data();
        stride_outs[i].bboxes = bbox_out->buffer.data();
        stride_outs[i].kps = (has_kps_ && kp_out) ? kp_out->buffer.data() : nullptr;
        stride_outs[i].stride = strides_[i];
        stride_outs[i].anchor_count = anchor_counts_[i];
        stride_outs[i].grid_h = input_size_ / strides_[i];
        stride_outs[i].grid_w = input_size_ / strides_[i];
    }
    
    // 使用 ScrfdDecoder 解码所有 stride 层的输出，执行 NMS
    // Use ScrfdDecoder to decode outputs from all stride layers, applying NMS
    if (results) {
        *results = ScrfdDecoder::Decode(stride_outs, conf_threshold_, iou_threshold_, max_count_);
    }
    
    return true;
}

void ScrfdModel::Unload() {
    if (backend_) {
        backend_->Unload();
    }
}

} // namespace face_rec
