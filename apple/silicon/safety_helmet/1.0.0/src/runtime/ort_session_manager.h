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

    bool Init(const std::string& model_path);
    
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
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_ORT_SESSION_MANAGER_H
