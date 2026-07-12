/**
 * @file inference_backend.h
 * @brief 推理后端抽象层 — 封装 ONNX Runtime 与 CoreML 两种推理引擎
 *        Inference backend abstraction — wraps ONNX Runtime and CoreML engines
 *
 * 定义了模型输入输出结构体、抽象接口 IInferenceBackend、
 * 执行提供者枚举（CPU/XNNPACK/CoreML/MPS），
 * 以及两个具体实现的前向声明。
 * Defines model input/output structs, the IInferenceBackend abstract interface,
 * execution provider enum (CPU/XNNPACK/CoreML/MPS),
 * and forward declarations for two concrete implementations.
 */
#ifndef FACE_RECOGNITION_INFERENCE_BACKEND_H
#define FACE_RECOGNITION_INFERENCE_BACKEND_H

#include <string>
#include <vector>
#include <memory>

namespace face_rec {

/**
 * @struct ModelInput
 * @brief 模型输入张量描述（零拷贝：仅持有指针）
 *        Model input tensor descriptor (zero-copy: holds pointers only)
 *
 * data 指向预分配的连续 float 缓冲区，
 * 不持有所有权，由调用方管理内存。
 * data points to a pre-allocated contiguous float buffer;
 * it does not own the memory — the caller manages it.
 */
struct ModelInput {
    const void* data;           ///< 归一化的 float 缓冲区 / Normalized flat float buffer
    size_t size;                ///< 元素个数 / Element count
    const int64_t* shape;       ///< 维度数组 / Dimensions array
    size_t shape_len;           ///< 维度数 / Number of dimensions
};

/**
 * @struct ModelOutput
 * @brief 模型输出张量（持有数据所有权）
 *        Model output tensor (owns the data)
 *
 * buffer 持有推理结果的 float 数据，由后端在 Run() 时填充。
 * buffer owns the float inference result data, filled by the backend during Run().
 */
struct ModelOutput {
    std::string name;           ///< 输出层名称 / Output layer name
    std::vector<float> buffer;  ///< 展平的输出数据 / Flattened output data
    int64_t shape[4];           ///< 输出张量形状（最多 4 维） / Output shape (max 4 dims)
};

/**
 * @class IInferenceBackend
 * @brief 推理后端抽象接口
 *        Abstract interface for inference backends
 *
 * 统一 Load → Run → Unload 生命周期，
 * 不同后端通过 PIMPL 模式隐藏实现细节。
 * Unifies the Load → Run → Unload lifecycle;
 * different backends use PIMPL to hide implementation details.
 */
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    /** @brief 加载模型文件 / Load the model file */
    virtual bool Load(const std::string& model_path) = 0;
    /** @brief 执行推理 / Run inference */
    virtual bool Run(const std::vector<ModelInput>& inputs,
                     std::vector<ModelOutput>* outputs) = 0;
    /** @brief 卸载模型，释放资源 / Unload the model, release resources */
    virtual void Unload() = 0;
};

/**
 * @enum OrtProvider
 * @brief ONNX Runtime 执行提供者枚举
 *        ONNX Runtime execution provider enum
 */
enum class OrtProvider {
    CPU,       ///< 默认 CPU 执行 / Default CPU execution
    XNNPACK,   ///< XNNPACK 加速（适用于 ARM CPU）/ XNNPACK acceleration (suitable for ARM CPUs)
    COREML,    ///< Apple CoreML（ANE+GPU+CPU 混合）/ Apple CoreML (ANE+GPU+CPU hybrid)
    MPS        ///< Metal Performance Shaders（Apple GPU）/ Metal Performance Shaders (Apple GPU)
};

/**
 * @class OnnxBackend
 * @brief ONNX Runtime 推理后端，支持多执行提供者
 *        ONNX Runtime inference backend, supporting multiple execution providers
 *
 * 使用 PIMPL 模式隐藏 Ort::Session 等 ONNX Runtime 内部类型。
 * Uses PIMPL to hide Ort::Session and other ONNX Runtime internals.
 */
class OnnxBackend : public IInferenceBackend {
public:
    /**
     * @brief 构造函数
     * @param provider 执行提供者（CPU/XNNPACK/CoreML/MPS）
     * @param intra_op_threads 算子内并行线程数 / intra-operator parallelism threads
     */
    OnnxBackend(OrtProvider provider, int intra_op_threads);
    ~OnnxBackend() override;

    bool Load(const std::string& model_path) override;
    bool Run(const std::vector<ModelInput>& inputs,
             std::vector<ModelOutput>* outputs) override;
    void Unload() override;

private:
    OrtProvider provider_;          ///< 执行提供者 / execution provider
    int intra_op_threads_;          ///< 算子内线程数 / intra-op thread count
    
    struct Impl;                    ///< PIMPL 前向声明 / forward declaration of PIMPL
    std::unique_ptr<Impl> impl_;    ///< PIMPL 实现指针 / pointer to PIMPL implementation
};

/**
 * @class CoreMLBackend
 * @brief Apple CoreML 推理后端（使用 MLModel API）
 *        Apple CoreML inference backend (using MLModel API)
 *
 * 编译 .mlmodelc 并使用 MLMultiArray 零拷贝包装输入数据。
 * Compiles .mlmodelc and wraps input data via MLMultiArray zero-copy.
 */
class CoreMLBackend : public IInferenceBackend {
public:
    CoreMLBackend();
    ~CoreMLBackend() override;

    bool Load(const std::string& model_path) override;
    bool Run(const std::vector<ModelInput>& inputs,
             std::vector<ModelOutput>* outputs) override;
    void Unload() override;

private:
    struct Impl;                    ///< PIMPL 前向声明（隐藏 MLModel* 等 ObjC 类型）
                                    ///< forward declaration of PIMPL (hides MLModel* etc. ObjC types)
    std::unique_ptr<Impl> impl_;    ///< PIMPL 实现指针 / pointer to PIMPL implementation
};

} // namespace face_rec

#endif // FACE_RECOGNITION_INFERENCE_BACKEND_H
