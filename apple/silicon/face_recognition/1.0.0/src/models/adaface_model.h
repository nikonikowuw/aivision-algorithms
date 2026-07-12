/**
 * @file adaface_model.h
 * @brief AdaFace 人脸识别模型类声明
 *        AdaFace face recognition model class declaration
 *
 * AdaFace 是一个基于 ArcFace/margin-based softmax 的高精度人脸识别模型，
 * 输入为 112×112 的对齐人脸图像，输出 512 维归一化嵌入向量。
 * AdaFace is a high-accuracy face recognition model based on ArcFace/margin-based softmax,
 * taking 112×112 aligned face images and producing 512-d normalized embedding vectors.
 */
#ifndef FACE_RECOGNITION_ADAFACE_MODEL_H
#define FACE_RECOGNITION_ADAFACE_MODEL_H

#include "model_interface.h"
#include <vector>
#include <string>

namespace face_rec {

/**
 * @class AdaFaceModel
 * @brief AdaFace 人脸识别模型，继承自 IModel 接口
 *        AdaFace face recognition model, implementing the IModel interface
 *
 * 输入：112×112 BGR 图像（由 AlignedFaceProcessor 对齐后传入）
 * 输出：512 维 L2 归一化的人脸嵌入向量
 * 后处理：将 Logits 层输出经过 L2 归一化得到最终 embedding
 *
 * Input: 112×112 BGR image (provided by AlignedFaceProcessor after alignment)
 * Output: 512-d L2-normalized face embedding vector
 * Postprocess: L2-normalize the logits layer output to obtain the final embedding
 */
class AdaFaceModel : public IModel {
public:
    /**
     * @brief 构造函数
     * @param recognition_threshold 人脸识别阈值（余弦相似度门槛） / recognition threshold (cosine similarity cutoff)
     */
    explicit AdaFaceModel(float recognition_threshold);
    ~AdaFaceModel() override = default;

    bool Load(const std::string& model_path,
              std::shared_ptr<IInferenceBackend> backend) override;

    /**
     * @brief 预处理接收对齐后的 112×112 人脸图像
     *        Preprocess receives the aligned 112x112 face Image
     */
    bool Preprocess(const Image& aligned_face, ModelInput* input,
                    image_utils::LetterboxInfo* info) override;

    bool Postprocess(const std::vector<ModelOutput>& raw_outputs,
                     const image_utils::LetterboxInfo& info,
                     std::vector<DetectedObject>* results) override;

    void Unload() override;

    /** @brief 获取输出层名称列表 / Get output layer names */
    const std::vector<std::string>& GetOutputNames() const { return output_names_; }

private:
    std::shared_ptr<IInferenceBackend> backend_;      ///< 推理后端 / inference backend
    
    // Pre-allocated buffers for zero-copy inputs
    std::vector<float> input_buffer_;     ///< 预分配的 float 输入缓冲区 [1,3,112,112] / pre-allocated float input buffer [1,3,112,112]
    std::vector<int64_t> input_shape_;    ///< 输入张量形状 / input tensor shape
    std::vector<std::string> output_names_;  ///< 输出层名称（固定为 "output"） / output layer name (fixed as "output")
};

} // namespace face_rec

#endif // FACE_RECOGNITION_ADAFACE_MODEL_H
