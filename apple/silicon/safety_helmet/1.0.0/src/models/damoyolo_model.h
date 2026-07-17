/**
 * @file damoyolo_model.h
 * @brief DAMO-YOLO-S model abstraction over ONNXRuntime.
 *
 * Thin wrapper around OrtSessionManager that provides the public model
 * interface consumed by SafetyHelmetPipeline. All session management
 * (execution provider selection, tensor allocation) is delegated to
 * OrtSessionManager.
 */

#ifndef SAFETY_HELMET_DAMOYOLO_MODEL_H
#define SAFETY_HELMET_DAMOYOLO_MODEL_H

#include "runtime/ort_session_manager.h"
#include <memory>
#include <vector>
#include <string>

namespace safety_helmet {

/**
 * @brief DAMO-YOLO-S object detection model.
 *
 * Lifecycle: Init() → Forward() (repeated) → destructor.
 */
class DamoYoloModel {
public:
    DamoYoloModel();
    ~DamoYoloModel() = default;

    /**
     * @brief Load the ONNX model from disk.
     *
     * On Apple Silicon, CoreML EP is automatically enabled by
     * OrtSessionManager::Init(). Falls back to CPU on other platforms.
     *
     * @param model_path  Absolute path to the .onnx file.
     * @return true on successful load and session creation.
     */
    bool Init(const std::string& model_path);

    /**
     * @brief Run inference on a preprocessed input tensor.
     *
     * @param input_data    Float tensor in CHW layout [3*target*target],
     *                      RGB order, values in [0,1].
     * @param input_shape   Tensor shape, typically [1, 3, 640, 640].
     * @param output_tensors [out] One or more float tensors from the model.
     * @param output_shapes  [out] Corresponding shapes.
     * @return true on successful forward pass.
     */
    bool Forward(const std::vector<float>& input_data,
                 const std::vector<int64_t>& input_shape,
                 std::vector<std::vector<float>>& output_tensors,
                 std::vector<std::vector<int64_t>>& output_shapes);

private:
    std::unique_ptr<OrtSessionManager> session_manager_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_DAMOYOLO_MODEL_H
