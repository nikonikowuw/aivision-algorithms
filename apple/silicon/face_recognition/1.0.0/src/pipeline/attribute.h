/**
 * attribute.h
 *
 * 属性提取模块 - Attribute Extraction Module
 *
 * 定义了人脸 / 人体属性提取器的抽象接口，以及用作优雅降级的空实现。
 * Defines the abstract interface for face/body attribute extractors and a
 * null implementation used for graceful degradation when no attribute model
 * is available.
 *
 * 未来可以在这里添加年龄估计、性别识别、表情识别等具体实现。
 * Future concrete implementations (age estimation, gender recognition,
 * expression analysis, etc.) can be added here.
 */

#ifndef FACE_RECOGNITION_ATTRIBUTE_H
#define FACE_RECOGNITION_ATTRIBUTE_H

#include "common/types.h"
#include <string>
#include <memory>

namespace face_rec {

class IInferenceBackend; // 前向声明推理后端 / Forward-declare the inference backend

/**
 * IAttributeExtractor - 属性提取器抽象接口
 * Abstract interface for attribute extraction
 *
 * 所有具体属性提取器（年龄、性别、表情等）应继承此接口。
 * All concrete attribute extractors (age, gender, expression, etc.)
 * should inherit from this interface.
 */
class IAttributeExtractor {
public:
    virtual ~IAttributeExtractor() = default;

    /** 返回提取器名称（如 "age", "gender"） / Return extractor name */
    virtual std::string Name() const = 0;

    /**
     * 加载模型并绑定推理后端
     * Load the model and bind to an inference backend
     *
     * @param model_path 模型文件路径 / Path to the model file
     * @param backend    推理后端共享指针 / Shared pointer to inference backend
     * @return true 加载成功 / true on success
     */
    virtual bool Load(const std::string& model_path,
                      std::shared_ptr<IInferenceBackend> backend) = 0;

    /**
     * 对 ROI 区域提取属性（返回 JSON 字符串）
     * Extract attributes from a region of interest (returns JSON string)
     *
     * @param roi 裁剪后的图像区域 / Cropped image region
     * @return 属性信息的 JSON 字符串 / JSON string of attribute data
     */
    virtual std::string Extract(const Image& roi) = 0;
};

/**
 * NullAttributeExtractor - 空属性提取器（空对象模式）
 * Null attribute extractor (Null Object Pattern)
 *
 * 当未配置属性模型时使用此实现，避免调用方进行空指针判断。
 * 始终返回空 JSON 对象 "{}"。
 * Used when no attribute model is configured; avoids null-pointer checks
 * at call sites by always returning the empty JSON object "{}".
 */
class NullAttributeExtractor : public IAttributeExtractor {
public:
    std::string Name() const override { return "null"; }
    bool Load(const std::string& model_path,
              std::shared_ptr<IInferenceBackend> backend) override {
        (void)model_path;
        (void)backend;
        return true;
    }
    std::string Extract(const Image& roi) override {
        (void)roi;
        return "{}";
    }
};

} // namespace face_rec

#endif // FACE_RECOGNITION_ATTRIBUTE_H
