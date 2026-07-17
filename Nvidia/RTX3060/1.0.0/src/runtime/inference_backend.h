/**
 * @file inference_backend.h
 * @brief GPU Face Recognition — Inference backend abstraction.
 *        Defines IInferenceBackend abstract interface, ModelInput/ModelOutput structs,
 *        and TrtBackend (TensorRT) concrete implementation.
 */

#ifndef GPU_FACE_RECOGNITION_INFERENCE_BACKEND_H
#define GPU_FACE_RECOGNITION_INFERENCE_BACKEND_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace face_rec {

/**
 * @struct ModelInput
 * @brief Model input tensor descriptor (zero-copy: holds pointers only).
 */
struct ModelInput {
    const void* data;           // Normalized float buffer
    size_t size;                // Element count
    const int64_t* shape;       // Dimensions array
    size_t shape_len;           // Number of dimensions
};

/**
 * @struct ModelOutput
 * @brief Model output tensor (owns the data).
 */
struct ModelOutput {
    std::string name;
    std::vector<float> buffer;
    int64_t shape[4];           // Output shape (max 4 dims)
};

/**
 * @class IInferenceBackend
 * @brief Abstract interface for inference backends.
 *        Unified Load → Run → Unload lifecycle.
 */
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual bool Load(const std::string& model_path) = 0;
    virtual bool Run(const std::vector<ModelInput>& inputs,
                     std::vector<ModelOutput>* outputs) = 0;
    virtual void Unload() = 0;
};

/**
 * @class TrtBackend
 * @brief TensorRT inference backend with dynamic batching (min=1, opt=8, max=16).
 *        Manages CUDA streams, device memory, and TensorRT engine/context lifecycle.
 */
class TrtBackend : public IInferenceBackend {
public:
    /**
     * @brief Constructor
     * @param cuda_device CUDA device ID (default: 0)
     * @param max_workspace Maximum workspace size in bytes (default: 2GB)
     */
    TrtBackend(int cuda_device = 0, size_t max_workspace = 2ULL * 1024 * 1024 * 1024);
    ~TrtBackend() override;

    bool Load(const std::string& engine_path) override;
    bool Run(const std::vector<ModelInput>& inputs,
             std::vector<ModelOutput>* outputs) override;
    void Unload() override;

    /** @brief Get the number of input bindings */
    int GetNumInputs() const;

    /** @brief Get the number of output bindings */
    int GetNumOutputs() const;

    /** @brief Get input binding dimensions */
    std::vector<int64_t> GetInputShape(int index = 0) const;

    /** @brief Get output binding dimensions */
    std::vector<int64_t> GetOutputShape(int index = 0) const;

    /** @brief Set actual batch size for dynamic shapes */
    void SetBatchSize(int batch_size);

private:
    int cuda_device_;
    size_t max_workspace_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_INFERENCE_BACKEND_H
