/**
 * @file model_interface.h
 * @brief 模型接口层 — 定义人脸检测/识别模型的抽象生命周期
 *        Model interface layer — defines the abstract lifecycle for face detection/recognition models
 *
 * 该文件声明了 IModel 抽象接口，统一所有模型子类（SCRFD 检测器、AdaFace 识别器）
 * 的加载、预处理、后处理、卸载四个阶段。
 * This file declares the IModel abstract interface, unifying the Load, Preprocess,
 * Postprocess, and Unload stages for all model subclasses (SCRFD detector, AdaFace recognizer).
 */
#ifndef FACE_RECOGNITION_MODEL_INTERFACE_H
#define FACE_RECOGNITION_MODEL_INTERFACE_H

#include "common/types.h"
#include "preprocess/image_utils.h"
#include "runtime/inference_backend.h"
#include <memory>
#include <string>
#include <vector>

namespace face_rec {

/**
 * @class IModel
 * @brief 模型抽象基类，定义推理全生命周期接口
 *        Abstract base class defining the full inference lifecycle interface
 *
 * 生命周期顺序：Load → Preprocess → [推理由 Backend 完成] → Postprocess → Unload
 * Lifecycle order: Load → Preprocess → [inference handled by Backend] → Postprocess → Unload
 */
class IModel {
public:
    virtual ~IModel() = default;
    
    /**
     * @brief 加载 ONNX/CoreML 模型文件并绑定推理后端
     *        Load the ONNX/CoreML model file and bind the inference backend
     * @param model_path 模型文件路径 / path to the model file
     * @param backend 推理后端共享指针 / shared pointer to the inference backend
     * @return true 表示加载成功 / true on success
     */
    virtual bool Load(const std::string& model_path,
                      std::shared_ptr<IInferenceBackend> backend) = 0;
                      
    /**
     * @brief 图像预处理：letterbox → 归一化 → CHW 排布
     *        Image preprocessing: letterbox → normalization → CHW layout
     * @param image 输入图像 / input image
     * @param input 输出模型输入结构体 / output model input struct
     * @param info 输出 letterbox 变换信息（用于反算坐标） / output letterbox transform info (for coordinate remapping)
     * @return true 表示预处理成功 / true on success
     */
    virtual bool Preprocess(const Image& image, ModelInput* input,
                            image_utils::LetterboxInfo* info) = 0;
                            
    /**
     * @brief 模型输出的后处理：解码、NMS、坐标映射等
     *        Postprocess raw model outputs: decode, NMS, coordinate remapping, etc.
     * @param raw_outputs 推理后端返回的原始输出张量 / raw output tensors from inference backend
     * @param info letterbox 信息（用于坐标从模型空间映射回原图） / letterbox info (for mapping coords from model space back to original image)
     * @param results 输出检测结果列表 / output list of detected objects
     * @return true 表示后处理成功 / true on success
     */
    virtual bool Postprocess(const std::vector<ModelOutput>& raw_outputs,
                             const image_utils::LetterboxInfo& info,
                             std::vector<DetectedObject>* results) = 0;
                             
    /**
     * @brief 卸载模型，释放所有资源
     *        Unload the model and release all resources
     */
    virtual void Unload() = 0;
};

} // namespace face_rec

#endif // FACE_RECOGNITION_MODEL_INTERFACE_H
