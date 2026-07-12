/**
 * @file scrfd_model.h
 * @brief SCRFD 人脸检测模型类声明
 *        SCRFD face detection model class declaration
 *
 * SCRFD (Sample and Computation Redistribution for Face Detection) 是一种高效的人脸检测器。
 * 该模型支持多尺度输出（640×640 用 5 stride, 320×320 用 3 stride），
 * 每个尺度输出 score / bbox / keypoints 三组张量。
 * SCRFD (Sample and Computation Redistribution for Face Detection) is an efficient face detector.
 * It supports multi-scale outputs (5 strides for 640×640, 3 strides for 320×320),
 * each scale producing score / bbox / keypoints tensors.
 */
#ifndef FACE_RECOGNITION_SCRFD_MODEL_H
#define FACE_RECOGNITION_SCRFD_MODEL_H

#include "model_interface.h"
#include <vector>
#include <string>

namespace face_rec {

/**
 * @class ScrfdModel
 * @brief SCRFD 人脸检测模型，继承自 IModel 接口
 *        SCRFD face detection model, implementing the IModel interface
 *
 * 核心特点：
 * - 预分配 input_buffer_ 实现零拷贝输入（避免每次推理动态分配）
 * - 构造函数根据 input_size 自动配置 stride 和 anchor 参数
 * - 后处理通过 ScrfdDecoder 解码各 stride 层的 anchor 输出
 *
 * Key features:
 * - Pre-allocated input_buffer_ for zero-copy input (avoids dynamic allocation per inference)
 * - Constructor auto-configures stride and anchor parameters based on input_size
 * - Post-processing decodes anchor outputs from each stride layer via ScrfdDecoder
 */
class ScrfdModel : public IModel {
public:
    /**
     * @brief 构造函数
     * @param input_size 模型输入尺寸（如 640 或 320） / model input size (e.g. 640 or 320)
     * @param conf_threshold 置信度阈值 / confidence threshold
     * @param iou_threshold NMS 的 IOU 阈值 / IOU threshold for NMS
     * @param max_count 最大检测人数 / maximum number of detections
     */
    explicit ScrfdModel(int input_size, float conf_threshold, float iou_threshold, int max_count, bool is_person_model = true);
    ~ScrfdModel() override = default;

    bool Load(const std::string& model_path,
              std::shared_ptr<IInferenceBackend> backend) override;

    bool Preprocess(const Image& image, ModelInput* input,
                    image_utils::LetterboxInfo* info) override;

    bool Postprocess(const std::vector<ModelOutput>& raw_outputs,
                     const image_utils::LetterboxInfo& info,
                     std::vector<DetectedObject>* results) override;

    void Unload() override;
    
    /** @brief 获取输出层名称列表（用于调试和信息） / Get output layer names (for debugging and info) */
    const std::vector<std::string>& GetOutputNames() const { return output_names_; }

private:
    int input_size_;          ///< 模型输入边长（正方形） / model input side length (square)
    float conf_threshold_;    ///< 检测置信度阈值 / detection confidence threshold
    float iou_threshold_;     ///< NMS 交并比阈值 / NMS intersection-over-union threshold
    int max_count_;           ///< 每帧最大检测人数 / maximum detections per frame
    
    // Normalization parameters for BlobFromImage
    float mean_ = 127.5f;     ///< 归一化均值 / normalization mean (127.5 for person, 0 for face)
    float std_ = 127.5f;      ///< 归一化标准差 / normalization std (127.5 for person, 1 for face)
    
    std::shared_ptr<IInferenceBackend> backend_;  ///< 推理后端 / inference backend
    
    // Pre-allocated buffers for zero-copy inputs
    std::vector<float> input_buffer_;     ///< 预分配的 float 输入缓冲区 [1,3,H,W] / pre-allocated float input buffer [1,3,H,W]
    std::vector<int64_t> input_shape_;    ///< 输入张量形状 / input tensor shape
    
    std::vector<std::string> output_names_;   ///< 输出层名称 / output layer names
    std::vector<int> strides_;                ///< 各 stride 层的采样步长 / stride values for each layer
    std::vector<int> anchor_counts_;          ///< 各 stride 层的 anchor 数量 / anchor count per stride layer
    bool has_kps_ = false;                    ///< 是否包含关键点输出 / whether keypoint outputs are present
};

} // namespace face_rec

#endif // FACE_RECOGNITION_SCRFD_MODEL_H
