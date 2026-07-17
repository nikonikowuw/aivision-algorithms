/**
 * @file ort_session_manager.cpp
 * @brief ONNX Runtime session management with CoreML→CPU fallback.
 *
 * ## Execution provider strategy
 *
 *   Apple Silicon:
 *     1. Build session with CoreML EP + COREML_FLAG_ENABLE_ON_SUBGRAPH.
 *     2. If CoreML EP fails to append → fall back to CPU EP at init time.
 *     3. If Run() fails with a CoreML error → rebuild session with CPU EP
 *        and retry once. The CPU session persists for subsequent frames.
 *
 *   Other platforms:
 *     CPU EP only.
 *
 * ## Threading
 *
 * IntraOpNumThreads and InterOpNumThreads = 1 because:
 *   - Engine manages its own thread pool for concurrent pipeline instances.
 *   - Each pipeline serialises frames via std::mutex.
 */

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
    // CPU memory allocator for both input and output tensors.
    // On Apple Silicon UMA, "CPU" memory is accessible to GPU/NPU via
    // the unified memory controller — no explicit transfer needed.
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
}

// ---------------------------------------------------------------------------
// Init() — first attempt with CoreML, fall back to CPU if CoreML fails.
// ---------------------------------------------------------------------------
bool OrtSessionManager::Init(const std::string& model_path) {
#ifdef __APPLE__
    // Try CoreML first; if load fails, fall back to CPU.
    if (BuildSession(model_path, /*enable_coreml=*/true)) {
        return true;
    }
    ALGO_LOG_WARN("CoreML session failed, falling back to CPU EP.");
    return BuildSession(model_path, /*enable_coreml=*/false);
#else
    return BuildSession(model_path, /*enable_coreml=*/false);
#endif
}

// ---------------------------------------------------------------------------
// BuildSession — shared by Init() and CoreML→CPU fallback.
// ---------------------------------------------------------------------------
bool OrtSessionManager::BuildSession(const std::string& model_path,
                                      bool enable_coreml) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableCpuMemArena();

        std::string provider_name = "CPU";

#ifdef __APPLE__
        if (enable_coreml) {
            // CoreML EP flags:
            //   0x001 — COREML_FLAG_ENABLE_ON_SUBGRAPH (partial delegation)
            //   0x002 — COREML_FLAG_ONLY_ENABLE_DEVICE_CPU_AND_GPU
            //           (disable ANE; GPU path has looser shape constraints
            //            and avoids the DAMO-YOLO output reshape mismatch)
            uint32_t coreml_flags = 0x001 | 0x002;

            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
                session_options, coreml_flags);
            if (status != nullptr) {
                Ort::GetApi().ReleaseStatus(status);
                ALGO_LOG_WARN("Failed to enable CoreML EP.");
            } else {
                provider_name = "CoreML";
                ALGO_LOG_INFO("CoreML Execution Provider enabled.");
            }
        }
#endif

        ALGO_LOG_INFO("Loading ONNX model: %s using EP: %s",
                      model_path.c_str(), provider_name.c_str());

        session_ = std::make_unique<Ort::Session>(
            env_, model_path.c_str(), session_options);

        CacheNodeNames();
        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("OrtSessionManager BuildSession failed: %s", e.what());
        session_.reset();
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("OrtSessionManager BuildSession failed: unknown exception");
        session_.reset();
        return false;
    }
}

// ---------------------------------------------------------------------------
// CacheNodeNames — extract input/output tensor names from session once.
// ---------------------------------------------------------------------------
void OrtSessionManager::CacheNodeNames() {
    if (!session_) return;

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session_->GetInputCount();
    input_names_.resize(num_inputs);
    input_node_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_allocated = session_->GetInputNameAllocated(i, allocator);
        input_names_[i] = name_allocated.get();
        input_node_names_[i] = input_names_[i].c_str();
    }

    size_t num_outputs = session_->GetOutputCount();
    output_names_.resize(num_outputs);
    output_node_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_allocated = session_->GetOutputNameAllocated(i, allocator);
        output_names_[i] = name_allocated.get();
        output_node_names_[i] = output_names_[i].c_str();
    }
}

// ---------------------------------------------------------------------------
// Run() — inference with automatic CoreML→CPU fallback on failure.
// ---------------------------------------------------------------------------
bool OrtSessionManager::Run(const std::vector<float>& input_data,
                            const std::vector<int64_t>& input_shape,
                            std::vector<std::vector<float>>& output_tensors,
                            std::vector<std::vector<int64_t>>& output_shapes) {
    if (!session_) {
        ALGO_LOG_ERROR("OrtSessionManager Run: session not initialized");
        return false;
    }

    try {
        // Validate shape ↔ data size consistency.
        size_t total_elements = 1;
        for (auto dim : input_shape) total_elements *= dim;
        if (total_elements != input_data.size()) {
            ALGO_LOG_ERROR("Input shape elements (%zu) != data size (%zu)",
                           total_elements, input_data.size());
            return false;
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

        auto ort_outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_node_names_.data(), &input_tensor, 1,
            output_node_names_.data(), output_node_names_.size());

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
        ALGO_LOG_ERROR("OrtSessionManager Run failed: unknown exception");
        return false;
    }
}

} // namespace safety_helmet
