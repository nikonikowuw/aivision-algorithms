#ifndef SAFETY_HELMET_DAMOYOLO_MODEL_H
#define SAFETY_HELMET_DAMOYOLO_MODEL_H

#include "runtime/ort_session_manager.h"
#include <memory>
#include <vector>
#include <string>

namespace safety_helmet {

class DamoYoloModel {
public:
    DamoYoloModel();
    ~DamoYoloModel() = default;

    bool Init(const std::string& model_path);

    bool Forward(const std::vector<float>& input_data,
                 const std::vector<int64_t>& input_shape,
                 std::vector<std::vector<float>>& output_tensors,
                 std::vector<std::vector<int64_t>>& output_shapes);

private:
    std::unique_ptr<OrtSessionManager> session_manager_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_DAMOYOLO_MODEL_H
