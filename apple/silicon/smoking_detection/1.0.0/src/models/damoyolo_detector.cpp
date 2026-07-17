// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - DAMO-YOLO detector implementation.

#include "damoyolo_detector.h"

#include <utility>

#include "../postprocess/damoyolo_decoder.h"
#include "../postprocess/nms.h"

namespace smoking {

DamoYoloDetector::DamoYoloDetector(
    std::unique_ptr<IInferenceBackend> backend)
    : backend_(std::move(backend)) {}

DamoYoloDetector::~DamoYoloDetector() { Unload(); }

bool DamoYoloDetector::Load(const ModelSpec& spec, std::string& error) {
    error.clear();
    if (!backend_) {
        error = "inference backend is not configured";
        return false;
    }
    spec_ = spec;
    loaded_ = backend_->Load(spec_, error);
    return loaded_;
}

ErrorCode DamoYoloDetector::Detect(uint64_t prepared_pixel_buffer,
                                   const ImageTransform& transform,
                                   std::vector<Detection>& detections,
                                   std::string& error) {
    detections.clear();
    error.clear();
    if (!loaded_ || prepared_pixel_buffer == 0) {
        error = "detector is not loaded or prepared input is null";
        return ErrorCode::kInvalidParam;
    }

    InferenceTensor tensor;
    if (!backend_->Run(prepared_pixel_buffer, tensor, error)) {
        return ErrorCode::kInferenceFailed;
    }
    if (tensor.values.empty() || tensor.values.size() %
            DamoOutputParser::kFieldsPerDetection != 0) {
        error = "DAMO output element count is not divisible by six";
        return ErrorCode::kPostprocessFailed;
    }

    const int32_t count = static_cast<int32_t>(
        tensor.values.size() / DamoOutputParser::kFieldsPerDetection);
    auto decoded = DamoOutputParser::Parse(tensor.values.data(), count,
                                           spec_.input_width,
                                           spec_.input_height, transform);
    decoded = FilterByConfidence(decoded, spec_.conf_threshold);
    SortByConfidence(decoded);
    const auto nms = ApplyNMS(decoded, spec_.nms_threshold);
    detections.reserve(nms.kept_indices.size());
    for (const int32_t index : nms.kept_indices) {
        if (index >= 0 && static_cast<size_t>(index) < decoded.size()) {
            detections.push_back(decoded[static_cast<size_t>(index)]);
        }
    }
    return ErrorCode::kSuccess;
}

void DamoYoloDetector::Unload() {
    backend_.reset();
    loaded_ = false;
}

}  // namespace smoking
