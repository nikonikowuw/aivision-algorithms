#include "ort_session_manager.h"
#include "common/logger.h"
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#include <cpu_provider_factory.h>
#endif
#include <cstring>
#include <algorithm>

namespace safety_helmet {

OrtSessionManager::OrtSessionManager()
    : env_(ORT_LOGGING_LEVEL_WARNING, "OrtSessionManager"),
      memory_info_(nullptr) {
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

bool OrtSessionManager::Init(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableCpuMemArena();

        std::string provider_name = "CPU";

#ifdef __APPLE__
        uint32_t coreml_flags = 0; // Use default flag
        // Allow subgraph delegation
        coreml_flags |= 0x001; // COREML_FLAG_ENABLE_ON_SUBGRAPH
        
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags);
        if (status != nullptr) {
            Ort::GetApi().ReleaseStatus(status);
            ALGO_LOG_WARN("Failed to enable CoreML EP, falling back to CPU.");
        } else {
            provider_name = "CoreML";
            ALGO_LOG_INFO("CoreML Execution Provider successfully enabled.");
        }
#else
        ALGO_LOG_WARN("CoreML is only supported on Apple platforms. Running on CPU.");
#endif

        ALGO_LOG_INFO("Loading ONNX model: %s using EP: %s", model_path.c_str(), provider_name.c_str());

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        // Get Input Nodes
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session_->GetInputCount();
        input_names_.resize(num_inputs);
        input_node_names_.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_allocated = session_->GetInputNameAllocated(i, allocator);
            input_names_[i] = name_allocated.get();
            input_node_names_[i] = input_names_[i].c_str();
        }

        // Get Output Nodes
        size_t num_outputs = session_->GetOutputCount();
        output_names_.resize(num_outputs);
        output_node_names_.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_allocated = session_->GetOutputNameAllocated(i, allocator);
            output_names_[i] = name_allocated.get();
            output_node_names_[i] = output_names_[i].c_str();
        }

        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("OrtSessionManager Init failed: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("OrtSessionManager Init failed with unknown exception");
        return false;
    }
}

bool OrtSessionManager::Run(const std::vector<float>& input_data,
                            const std::vector<int64_t>& input_shape,
                            std::vector<std::vector<float>>& output_tensors,
                            std::vector<std::vector<int64_t>>& output_shapes) {
    if (!session_) {
        ALGO_LOG_ERROR("OrtSessionManager Run failed: session not initialized");
        return false;
    }

    try {
        size_t total_elements = 1;
        for (auto dim : input_shape) total_elements *= dim;

        if (total_elements != input_data.size()) {
            ALGO_LOG_ERROR("Input shape elements count (%zu) does not match input data size (%zu)",
                           total_elements, input_data.size());
            return false;
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto ort_outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(),
            &input_tensor,
            1,
            output_node_names_.data(),
            output_node_names_.size()
        );

        output_tensors.resize(ort_outputs.size());
        output_shapes.resize(ort_outputs.size());

        for (size_t i = 0; i < ort_outputs.size(); ++i) {
            auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            size_t num_elements = type_info.GetElementCount();
            const float* raw_data = ort_outputs[i].GetTensorData<float>();

            output_tensors[i].assign(raw_data, raw_data + num_elements);
            output_shapes[i] = shape;
        }

        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("OrtSessionManager Run failed: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("OrtSessionManager Run failed with unknown exception");
        return false;
    }
}

} // namespace safety_helmet
