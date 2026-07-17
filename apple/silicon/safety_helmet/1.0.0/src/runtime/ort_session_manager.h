/**
 * @file ort_session_manager.h
 * @brief ONNX Runtime session management — model loading, EP selection, inference.
 *
 * On Apple Silicon, the CoreML Execution Provider (ANE/GPU) is mandatory.
 * On other platforms, the CPU EP is used.
 *
 * Thread count is set to 1 to avoid contention with the Engine's thread pool.
 */

#ifndef SAFETY_HELMET_ORT_SESSION_MANAGER_H
#define SAFETY_HELMET_ORT_SESSION_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace safety_helmet {

class OrtSessionManager {
public:
    OrtSessionManager();
    ~OrtSessionManager() = default;

    /**
     * @brief Load an ONNX model and configure execution providers.
     *
     * On Apple Silicon, enables CoreML EP first. If CoreML fails at load
     * time, falls back to CPU EP immediately.
     *
     * Thread count is set to 1 to avoid contention with the Engine's
     * thread pool.
     *
     * @param model_path  Absolute path to the .onnx model file.
     * @return true on success.
     */
    bool Init(const std::string& model_path);
    
    /**
     * @brief Run inference with the given input tensor.
     *
     * @param input_data      Flat float vector in CHW layout.
     * @param input_shape     Dimensions (e.g. [1, 3, 640, 640]).
     * @param output_tensors  [out] Vector of float tensors (one per output node).
     * @param output_shapes   [out] Corresponding tensor shapes.
     * @return true on success.
     */
    bool Run(const std::vector<float>& input_data,
             const std::vector<int64_t>& input_shape,
             std::vector<std::vector<float>>& output_tensors,
             std::vector<std::vector<int64_t>>& output_shapes);

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;

    /**
     * @brief Build an ORT session with or without CoreML EP.
     *
     * @param model_path     Absolute path to ONNX model.
     * @param enable_coreml  If true, append CoreML EP; false = CPU only.
     * @return true on success.
     */
    bool BuildSession(const std::string& model_path, bool enable_coreml);

    /** @brief Cache input/output node names from the current session. */
    void CacheNodeNames();
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_ORT_SESSION_MANAGER_H
