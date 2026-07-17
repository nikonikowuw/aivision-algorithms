/**
 * @file algorithm_context.h
 * @brief GPU Face Recognition — Pipeline orchestrator.
 *        Central scheduling class that chains all pipeline modules:
 *        Input → Person Detection → Tracking → Face Detection →
 *        IoU Association → Face Alignment → Feature Extraction →
 *        Gallery Matching → JSON Result Output
 * @module Pipeline Layer
 */

#ifndef GPU_FACE_RECOGNITION_ALGORITHM_CONTEXT_H
#define GPU_FACE_RECOGNITION_ALGORITHM_CONTEXT_H

#include "common/types.h"
#include "common/config.h"
#include "common/logger.h"
#include "preprocess/image_utils.h"
#include "postprocess/coordinate.h"
#include "postprocess/yolo_decoder.h"
#include "postprocess/retina_decoder.h"
#include "postprocess/face_aligner.h"
#include "pipeline/tracker.h"
#include "pipeline/face_index.h"
#include "pipeline/batch_collector.h"
#include "pipeline/thread_pool.h"
#include "models/model_interface.h"
#include "models/yolo11_model.h"
#include "models/retinaface_model.h"
#include "models/adaface_model.h"
#include "runtime/inference_backend.h"
#include "algo/abi_contract.h"

#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace face_rec {

/**
 * @class AlgorithmContext
 * @brief GPU face recognition pipeline orchestrator.
 *
 * Lifecycle:
 *   1. Initialize(config_json)  — 4-stage init
 *   2. Infer(input, ctx, result) — Per-frame inference
 *   3. UpdateFaceLibrary(json)   — Hot-update face gallery
 *   4. Destroy()                 — Release all resources
 */
class AlgorithmContext {
public:
    AlgorithmContext() = default;
    ~AlgorithmContext();

    bool Initialize(const char* config_json);
    int Infer(const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result);
    int UpdateFaceLibrary(const char* face_library_json);
    void Destroy();

private:
    Config config_;
    bool initialized_ = false;

    // ---- Inference Backends (TensorRT) ----
    std::shared_ptr<IInferenceBackend> person_backend_;
    std::shared_ptr<IInferenceBackend> face_backend_;
    std::shared_ptr<IInferenceBackend> face_rec_backend_;

    // ---- Model Instances ----
    std::unique_ptr<Yolo11Model> person_detector_;
    std::unique_ptr<RetinaFaceModel> face_detector_;
    std::unique_ptr<AdaFaceModel> face_recognizer_;

    // ---- Pipeline Components ----
    std::unique_ptr<IFaceAligner> aligner_;
    std::unique_ptr<ITracker> tracker_;
    FaceIndex face_index_;
    std::unique_ptr<ThreadPool> thread_pool_;

    // ---- Synchronization & Timing ----
    std::mutex infer_mutex_;
    PipelineTiming timing_;
    int frame_count_ = 0;

    // Reusable buffers
    std::vector<uint8_t> bgr_buffer_;
    std::vector<uint8_t> aligned_face_buffer_;
    std::vector<ModelOutput> person_raw_outputs_;
    std::vector<ModelOutput> face_raw_outputs_;
    std::vector<ModelOutput> recognition_raw_outputs_;

    // ---- Internal pipeline methods ----
    int InferFrame(const uint8_t* bgr_data, int width, int height,
                   const char* context_json, infer_result_t* result);
    std::string BuildResultJson(const std::vector<DetectedObject>& faces, int img_w, int img_h);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_ALGORITHM_CONTEXT_H
