/**
 * @file damoyolo_model.cpp
 * @brief DAMO-YOLO-S model — thin delegation to OrtSessionManager.
 */

#include "damoyolo_model.h"

namespace safety_helmet {

DamoYoloModel::DamoYoloModel()
    : session_manager_(std::make_unique<OrtSessionManager>()) {}

bool DamoYoloModel::Init(const std::string& model_path) {
    return session_manager_->Init(model_path);
}

bool DamoYoloModel::Forward(const std::vector<float>& input_data,
                            const std::vector<int64_t>& input_shape,
                            std::vector<std::vector<float>>& output_tensors,
                            std::vector<std::vector<int64_t>>& output_shapes) {
    return session_manager_->Run(input_data, input_shape, output_tensors, output_shapes);
}

} // namespace safety_helmet
