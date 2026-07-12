/**
 * algorithm_context.h
 *
 * 算法上下文 / 管线编排器 - Algorithm Context / Pipeline Orchestrator
 *
 * AlgorithmContext 是整个识别管线的核心调度类，串联所有模块：
 *   输入 → 人体检测 → 跟踪 → 人头 ROI 裁切 → 人脸检测 → 人脸对齐 →
 *   人脸特征提取 → 底库检索 → JSON 结果输出
 *
 * AlgorithmContext is the central orchestrator of the entire recognition
 * pipeline, chaining all modules together:
 *   Input → Body Detection → Tracking → Head ROI Crop → Face Detection →
 *   Face Alignment → Feature Extraction → Database Search → JSON Output
 *
 * 初始化分为 4 个阶段：
 *   1. 加载配置和日志 / Load config & logger
 *   2. 创建推理后端 / Create inference backends
 *   3. 加载模型 / Load all models
 *   4. 创建管线组件 / Create pipeline components
 *
 * Initialization proceeds in 4 stages:
 *   1. Load configuration and initialize logger
 *   2. Create inference backends (CoreML / ONNX)
 *   3. Load all models (SCRFD person, SCRFD face, AdaFace)
 *   4. Create pipeline components (aligner, tracker, face index, thread pool, etc.)
 */

#ifndef FACE_RECOGNITION_ALGORITHM_CONTEXT_H
#define FACE_RECOGNITION_ALGORITHM_CONTEXT_H

#include "common/types.h"
#include "common/config.h"
#include "common/logger.h"
#include "preprocess/image_utils.h"
#include "postprocess/coordinate.h"
#include "postprocess/scrfd_decoder.h"
#include "postprocess/face_aligner.h"
#include "pipeline/tracker.h"
#include "pipeline/face_index.h"
#include "pipeline/thread_pool.h"
#include "pipeline/attribute.h"
#include "models/model_interface.h"
#include "models/scrfd_model.h"
#include "models/adaface_model.h"
#include "runtime/inference_backend.h"
#include "algo/abi_contract.h"

#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace face_rec {

/**
 * AlgorithmContext - 人脸识别管线调度器
 * Face recognition pipeline orchestrator
 *
 * 生命周期 / Lifecycle:
 *   1. Initialize(config_json)  — 初始化所有组件 / Initialize all components
 *   2. Infer(input, ctx, result) — 每帧推理 / Run inference on each frame
 *   3. UpdateFaceLibrary(json)   — 更新底库 / Update face database
 *   4. Destroy()                 — 释放资源 / Release all resources
 */
class AlgorithmContext {
public:
    AlgorithmContext() = default;
    ~AlgorithmContext();

    /**
     * 初始化管线（4 阶段）
     * Initialize the pipeline (4 stages)
     *
     * @param config_json JSON 配置字符串 / JSON configuration string
     * @return true 初始化成功 / true on success
     */
    bool Initialize(const char* config_json);

    /**
     * 执行一帧推理（完整管线）
     * Run inference on one frame (full pipeline)
     *
     * @param input       输入帧（NV12 或 BGR24） / Input frame (NV12 or BGR24)
     * @param context_json 额外上下文参数 / Additional context parameters
     * @param result      输出结果结构 / Output result struct
     * @return 0 成功 / 0 on success, negative on error
     */
    int Infer(const hw_buffer_desc_t* input, const char* context_json,
              infer_result_t* result);

    /** 释放所有资源 / Release all resources */
    void Destroy();

    /**
     * 更新人脸底库（JSON 格式）
     * Update the face database (JSON format)
     *
     * @param face_library_json 底库 JSON 字符串 / Face library JSON string
     * @return 0 成功 / 0 on success, negative on error
     */
    int UpdateFaceLibrary(const char* face_library_json);

private:
    Config config_;                                // 运行时配置 / Runtime configuration
    bool initialized_ = false;                     // 初始化成功标志 / Initialization success flag

    // ---- 推理后端 / Inference Backends ----
    std::shared_ptr<IInferenceBackend> person_backend_;    // 人体检测后端 / Body detection backend
    std::shared_ptr<IInferenceBackend> face_backend_;      // 人脸检测后端 / Face detection backend
    std::shared_ptr<IInferenceBackend> face_rec_backend_;   // 人脸识别后端 / Face recognition backend

    // ---- 模型实例 / Model Instances ----
    std::unique_ptr<ScrfdModel> person_detector_;   // 人体检测模型 / Person (body) detection model
    std::unique_ptr<ScrfdModel> face_detector_;      // 人脸检测模型 / Face detection model
    std::unique_ptr<AdaFaceModel> face_recognizer_;   // 人脸特征提取模型 / Face feature extraction model

    // ---- 管线组件 / Pipeline Components ----
    std::unique_ptr<IFaceAligner> aligner_;       // 人脸对齐器 / Face aligner (InsightFace)
    std::unique_ptr<ITracker> tracker_;           // 目标跟踪器 / Object tracker (ByteTracker)
    FaceIndex face_index_;                        // 人脸底库索引 / Face database index
    std::unique_ptr<ThreadPool> thread_pool_;     // 线程池 / Thread pool for parallel tasks
    std::unique_ptr<IAttributeExtractor> body_attr_;  // 人体属性提取器 / Body attribute extractor
    std::unique_ptr<IAttributeExtractor> face_attr_;  // 人脸属性提取器 / Face attribute extractor

    // ---- 同步与时序统计 / Synchronization & Timing ----
    std::mutex infer_mutex_;                      // 推理互斥锁（串行化 Infer） / Serialize Infer calls
    PipelineTiming timing_;                       // 各阶段计时统计 / Per-stage timing stats
    int frame_count_ = 0;                         // 帧计数器 / Frame counter
    std::vector<uint8_t> decoded_bgr_buffer_;     // NV12→BGR 解码缓存 / NV12-to-BGR decode buffer
};

} // namespace face_rec

#endif // FACE_RECOGNITION_ALGORITHM_CONTEXT_H
