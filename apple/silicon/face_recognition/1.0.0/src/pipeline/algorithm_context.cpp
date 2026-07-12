/**
 * algorithm_context.cpp
 *
 * 算法上下文 / 管线编排器实现 - Algorithm Context / Pipeline Orchestrator Implementation
 *
 * 实现了完整的人脸识别推理管线编排：
 *   Initialize:  4 阶段初始化（配置 → 后端 → 模型 → 组件）
 *   Infer:      输入帧 → 人体检测 → 跟踪 → ROI 裁切 → 人脸检测 → 对齐 → 特征提取 → 底库检索 → JSON
 *   UpdateFaceLibrary: 解析 JSON 底库 → 原子更新双缓冲快照
 *   Destroy:    逆序释放全部资源
 *
 * Implements the full face recognition pipeline orchestration.
 */

#include "algorithm_context.h"
#include "common/logger.h"
#include "common/config.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace face_rec {

namespace {

/**
 * 从 JSON 字符串中提取指定字段的字符串值（简易 JSON 解析器，无外部依赖）
 * Extract a string field value from a JSON string (lightweight parser, no external dependency)
 *
 * @param json         原始 JSON 字符串 / Raw JSON string
 * @param field_name   字段名（如 "id", "name"） / Field name (e.g. "id", "name")
 * @param search_start 搜索起始位置 / Search start offset
 * @param out          输出字符串 / Output string
 * @param out_end      字段结束位置（供后续解析使用） / End offset of the field (for chained parsing)
 * @return true 找到并提取成功 / true if found and extracted
 */
bool ExtractJsonStringField(const std::string& json, const std::string& field_name, size_t search_start, std::string* out, size_t* out_end) {
    std::string search_key = "\"" + field_name + "\"";
    size_t key_pos = json.find(search_key, search_start);
    if (key_pos == std::string::npos) return false;

    size_t colon_pos = json.find(':', key_pos);
    if (colon_pos == std::string::npos) return false;

    size_t start_quote = json.find('"', colon_pos);
    if (start_quote == std::string::npos) return false;

    size_t end_quote = json.find('"', start_quote + 1);
    if (end_quote == std::string::npos) return false;

    *out = json.substr(start_quote + 1, end_quote - start_quote - 1);
    if (out_end) *out_end = end_quote;
    return true;
}

/**
 * 从 JSON 数组中解析 float 列表（如 "embedding": [...])
 * Parse a float array from a JSON array (e.g. "embedding": [...])
 *
 * @param json       原始 JSON 字符串 / Raw JSON string
 * @param start_pos  搜索起始位置 / Search start offset
 * @param end_pos    数组结束位置 / End offset of the array
 * @return 解析后的 float 向量 / Parsed float vector
 */
std::vector<float> ParseFloatArray(const std::string& json, size_t start_pos, size_t* end_pos) {
    std::vector<float> arr;
    size_t open_bracket = json.find('[', start_pos);
    if (open_bracket == std::string::npos) return arr;

    size_t close_bracket = json.find(']', open_bracket);
    if (close_bracket == std::string::npos) return arr;

    // 提取方括号内的逗号分隔数值 / Extract comma-separated values inside brackets
    std::string contents = json.substr(open_bracket + 1, close_bracket - open_bracket - 1);
    std::stringstream ss(contents);
    std::string val_str;
    while (std::getline(ss, val_str, ',')) {
        try {
            arr.push_back(std::stof(val_str));
        } catch (...) {
            // 忽略解析错误 / Ignore parse errors for individual values
        }
    }

    if (end_pos) *end_pos = close_bracket;
    return arr;
}

/**
 * 转义字符串中的特殊字符，使其安全嵌入 JSON 字符串
 * Escape special characters in a string for safe JSON embedding
 *
 * 转义规则 / Escape rules:
 *   " → \",  \ → \\,  \b → \b,  \f → \f,  \n → \n,  \r → \r,  \t → \t
 *   控制字符 (< 0x20) 转为 \uXXXX
 */
std::string EscapeJsonString(const std::string& str) {
    std::string s;
    s.reserve(str.size() + 8);
    for (const auto& c : str) {
        unsigned char uc = static_cast<unsigned char>(c);
        switch (uc) {
            case '"':  s += "\\\""; break;
            case '\\': s += "\\\\"; break;
            case '\b': s += "\\b";  break;
            case '\f': s += "\\f";  break;
            case '\n': s += "\\n";  break;
            case '\r': s += "\\r";  break;
            case '\t': s += "\\t";  break;
            default:
                // 控制字符转为 Unicode 转义 / Control chars → Unicode escape
                if (uc < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", uc);
                    s += buf;
                } else {
                    s += c;
                }
                break;
        }
    }
    return s;
}

/**
 * 工厂函数：根据模型路径创建合适的推理后端
 * Factory function: create the appropriate inference backend based on model path
 *
 * 规则 / Rules:
 *   .mlpackage → CoreMLBackend（Apple Silicon GPU/ANE 加速）
 *   "person" 关键词路径 → OnnxBackend with XNNPACK provider
 *   默认 → OnnxBackend with CPU provider
 */
std::shared_ptr<IInferenceBackend> CreateBackend(const std::string& model_path, int intra_op_threads) {
    if (model_path.size() >= 10 && model_path.compare(model_path.size() - 10, 10, ".mlpackage") == 0) {
        return std::make_shared<CoreMLBackend>();
    }
    // Default fallback to ONNX
    if (model_path.find("person") != std::string::npos) {
        return std::make_shared<OnnxBackend>(OrtProvider::XNNPACK, intra_op_threads);
    }
    return std::make_shared<OnnxBackend>(OrtProvider::CPU, intra_op_threads);
}

} // namespace

AlgorithmContext::~AlgorithmContext() {
    Destroy();
}

/**
 * 释放所有资源（逆序销毁，先释放模型再释放后端）
 * Release all resources (reverse order: destroy models before backends)
 */
void AlgorithmContext::Destroy() {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    // 逆序释放：模型 → 跟踪器 → 对齐器 → 后端
    // Reverse order: models → tracker → aligner → backends
    person_detector_.reset();
    face_detector_.reset();
    face_recognizer_.reset();
    person_backend_.reset();
    face_backend_.reset();
    face_rec_backend_.reset();
    tracker_.reset();
    aligner_.reset();
    thread_pool_.reset();
    body_attr_.reset();
    face_attr_.reset();
}

/**
 * 管线初始化（4 阶段）
 * Pipeline initialization (4 stages)
 *
 * Stage 1: 加载配置与日志 / Load configuration & logger
 * Stage 2: 创建推理后端 / Create inference backends (CoreML or ONNX)
 * Stage 3: 加载并验证各模型 / Load and validate all models
 * Stage 4: 创建管线组件 / Create pipeline components (aligner, tracker, face index, thread pool, attrs)
 */
bool AlgorithmContext::Initialize(const char* config_json) {
    try {
        // ---- Stage 1: 加载配置与日志 ----
        // Load configuration and initialize logger
        config_ = Config::LoadConfig(config_json);
        Logger::Instance().Initialize(config_.log_levels);

        ALGO_LOGI(CONFIG, "Initializing M1 Pro Face Recognition Pipeline...");

        // 拼接模型文件的绝对路径 / Build absolute paths for model files
        std::string person_path = JoinPath(config_.package_dir, config_.scrfd_person_model_path);
        std::string face_path = JoinPath(config_.package_dir, config_.scrfd_500m_model_path);
        std::string rec_path = JoinPath(config_.package_dir, config_.adaface_model_path);

        // ---- Stage 2: 创建推理后端 ----
        // Create inference backends based on model file extensions
        person_backend_ = CreateBackend(person_path, config_.ort_intra_op_threads);
        face_backend_ = CreateBackend(face_path, config_.ort_intra_op_threads);
        face_rec_backend_ = CreateBackend(rec_path, config_.ort_intra_op_threads);

        // ---- Stage 3: 创建并加载模型 ----
        // Create and load all models
        person_detector_ = std::make_unique<ScrfdModel>(640, config_.person_conf_thres, config_.nms_iou_thres, config_.max_persons, true);
        face_detector_ = std::make_unique<ScrfdModel>(160, config_.face_conf_thres, config_.nms_iou_thres, config_.max_faces, false);
        face_recognizer_ = std::make_unique<AdaFaceModel>(config_.recognition_threshold);

        // 加载并校验人体检测模型 / Load and validate person detection model
        if (!person_detector_->Load(person_path, person_backend_)) {
            ALGO_LOGE(CONFIG, "Failed to load Person detection model: %s", person_path.c_str());
            return false;
        }
        // 加载并校验人脸检测模型 / Load and validate face detection model
        if (!face_detector_->Load(face_path, face_backend_)) {
            ALGO_LOGE(CONFIG, "Failed to load Face detection model: %s", face_path.c_str());
            return false;
        }
        // 加载并校验人脸识别模型 / Load and validate face recognition model
        if (!face_recognizer_->Load(rec_path, face_rec_backend_)) {
            ALGO_LOGE(CONFIG, "Failed to load AdaFace model: %s", rec_path.c_str());
            return false;
        }

        // ---- Stage 4: 创建管线组件 ----
        // Create pipeline components
        aligner_ = std::make_unique<InsightFaceAligner>();

        tracker_ = std::make_unique<ByteTracker>();
        tracker_->Initialize(config_.tracker_iou_thres, config_.tracker_max_lost_frames);

        face_index_.Initialize(config_.recognition_threshold);
        thread_pool_ = std::make_unique<ThreadPool>(config_.thread_pool_size);
        // 默认使用空属性提取器（优雅降级）/ Default to null attribute extractors (graceful deg.)
        body_attr_ = std::make_unique<NullAttributeExtractor>();
        face_attr_ = std::make_unique<NullAttributeExtractor>();

        initialized_ = true;
        ALGO_LOGI(CONFIG, "Face Recognition Pipeline initialized successfully.");
        return true;
    } catch (const std::exception& e) {
        ALGO_LOGE(CONFIG, "AlgorithmContext Initialization failed: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOGE(CONFIG, "AlgorithmContext Initialization failed with unknown exception");
        return false;
    }
}

/**
 * 更新人脸底库（JSON 格式解析 + 原子快照替换）
 * Update the face database (JSON parsing + atomic snapshot replacement)
 *
 * 输入 JSON 格式 / Input JSON format:
 * {
 *   "version": "v1.0",
 *   "identities": [
 *     {"id": "001", "name": "Alice", "embedding": [0.1, 0.2, ...]},
 *     ...
 *   ]
 * }
 *
 * 使用简易 JSON 解析器逐条解析，构造 KnownIdentity 列表后调用
 * face_index_.Update() 原子替换底库。
 * Uses a lightweight JSON parser to parse entries one by one, then calls
 * face_index_.Update() to atomically replace the entire database.
 */
int AlgorithmContext::UpdateFaceLibrary(const char* face_library_json) {
    if (!face_library_json) return -1;

    std::string json(face_library_json);

    // 提取底库版本号 / Extract database version
    std::string version = "unknown";
    size_t version_end = 0;
    ExtractJsonStringField(json, "version", 0, &version, &version_end);

    std::vector<KnownIdentity> parsed_identities;

    // 循环解析所有 identity 条目 / Loop-parsing all identity entries
    size_t search_pos = 0;
    while (true) {
        size_t id_pos = json.find("\"id\"", search_pos);
        if (id_pos == std::string::npos) break;

        std::string id, name;
        size_t id_end = 0, name_end = 0;
        // 提取 id 字段 / Extract id field
        if (!ExtractJsonStringField(json, "id", search_pos, &id, &id_end)) {
            break;
        }

        // 提取 name 字段 / Extract name field
        ExtractJsonStringField(json, "name", id_end, &name, &name_end);

        // 提取 embedding 数组 / Extract embedding array
        size_t emb_end = 0;
        std::vector<float> embedding = ParseFloatArray(json, std::max(id_end, name_end), &emb_end);

        // 校验：id 非空且特征维度为 512 / Validate: non-empty id and 512-d embedding
        if (!id.empty() && embedding.size() == 512) {
            KnownIdentity identity;
            identity.id = id;
            identity.name = name;
            identity.embedding = embedding;
            parsed_identities.push_back(identity);
        }

        // 移动到下一个身份记录的起始位置 / Advance to next identity entry
        search_pos = std::max({id_end, name_end, emb_end});
        if (search_pos == 0) {
            break;
        }
    }

    // 原子替换底库快照 / Atomically replace the database snapshot
    face_index_.Update([&](FaceIndex::Snapshot& snap) {
        snap.version = version;
        snap.identities = parsed_identities;
    });

    ALGO_LOGI(FACE_INDEX, "Face library updated. Version: %s, Identities count: %d",
              version.c_str(), static_cast<int>(parsed_identities.size()));

    return 0;
}

/**
 * 核心推理管线：输入帧 → JSON 结果
 * Core inference pipeline: input frame → JSON result
 *
 * 完整流程 / Complete flow:
 *   1. 解码输入帧（NV12 → BGR24 或直接封装） / Decode input frame
 *   2. 人体检测 / Body detection (SCRFD-640)
 *   3. 目标跟踪 / Tracking (ByteTracker IOU matching)
 *   4. 人头 ROI 裁切 / Head ROI cropping
 *   5. 人脸检测 / Face detection (SCRFD-320 on head ROI)
 *   6. 人脸对齐 / Face alignment (InsightFace 5-point align)
 *   7. 人脸特征提取 / Feature extraction (AdaFace 512-d embedding)
 *   8. 坐标映射回原图 / Map coordinates back to original frame
 *   9. 底库检索 / Database search (cosine similarity Top-K)
 *  10. 构造 JSON 输出 / Build JSON output
 */
int AlgorithmContext::Infer(const hw_buffer_desc_t* input, const char* context_json,
                            infer_result_t* result) {
    if (!input || !result) return -1;
    if (!initialized_) return -1;
    // TODO: context_json 用于传递场景 ID/session 上下文到底库检索（多场景隔离识别），当前版本尚未实现
    // TODO: context_json carries scene ID/session context for database search (multi-scene isolation); not yet implemented in this version

    // 串行化 Infer 调用 / Serialize Infer calls
    std::lock_guard<std::mutex> lock(infer_mutex_);

    timing_.total.Start();

    int frame_w = static_cast<int>(input->width);
    int frame_h = static_cast<int>(input->height);

    if (frame_w <= 0 || frame_h <= 0 || !input->data) {
        ALGO_LOGE(PIPELINE, "Infer failed: Invalid input frame parameters.");
        return -2;
    }

    // ---- Step 1: 帧解码 ----
    // Frame decoding: support NV12 and BGR24 pixel formats
    // 使用 types.h 中定义的 pixel_format::BGR24 / pixel_format::NV12

    Image frame_img;
    if (input->pixel_format == pixel_format::BGR24) {
        // BGR24 直接封装 / Directly wrap BGR24 memory
        frame_img = Image{reinterpret_cast<const uint8_t*>(input->data), frame_w, frame_h, 3, static_cast<int>(input->stride)};
    } else if (input->pixel_format == pixel_format::NV12) {
        // NV12 转 BGR24 / Convert NV12 to BGR24
        decoded_bgr_buffer_.resize(frame_w * frame_h * 3);
        image_utils::NV12ToBGR(reinterpret_cast<const uint8_t*>(input->data), frame_w, frame_h, decoded_bgr_buffer_.data());
        frame_img = Image{decoded_bgr_buffer_.data(), frame_w, frame_h, 3, frame_w * 3};
    } else {
        // 未知格式，尝试按 BGR24 处理 / Unknown format, fallback to BGR24
        frame_img = Image{reinterpret_cast<const uint8_t*>(input->data), frame_w, frame_h, 3, static_cast<int>(input->stride)};
    }

    // ---- Step 2: 人体检测预处理 ----
    // Body detection preprocessing
    timing_.body_detect.Start();
    ModelInput person_in;
    image_utils::LetterboxInfo person_letterbox;
    if (!person_detector_->Preprocess(frame_img, &person_in, &person_letterbox)) {
        ALGO_LOGE(PIPELINE, "Person detector preprocessing failed.");
        return -3;
    }

    // ---- Step 3: 人体检测推理 ----
    // Body detection inference
    std::vector<ModelOutput> person_raw_outs;
    std::vector<ModelInput> person_inputs = {person_in};
    if (!person_backend_->Run(person_inputs, &person_raw_outs)) {
        ALGO_LOGE(PIPELINE, "Person detector session run failed.");
        return -4;
    }

    // ---- Step 4: 人体检测后处理 ----
    // Body detection postprocessing (NMS, decode boxes)
    std::vector<DetectedObject> body_objects;
    if (!person_detector_->Postprocess(person_raw_outs, person_letterbox, &body_objects)) {
        ALGO_LOGE(PIPELINE, "Person detector postprocessing failed.");
        return -5;
    }
    timing_.body_detect.Stop();

    // 将人体检测结果从 letterbox 坐标系映射回原图坐标系
    // Map body detection results from letterbox coords to original frame coords
    CoordinateMapper::RestoreToOriginal(person_letterbox, frame_w, frame_h, &body_objects);

    // ---- Step 5: 人体目标跟踪 ----
    // Body object tracking (assign track IDs)
    timing_.tracker.Start();
    std::vector<DetectedObject> tracked_bodies;
    if (config_.enable_tracker && tracker_) {
        tracked_bodies = tracker_->Update(body_objects);
    } else {
        tracked_bodies = body_objects;
        for (auto& body : tracked_bodies) {
            body.track_id = -1;  // 禁用跟踪时 track_id 置为 -1 / Set to -1 when tracking is disabled
        }
    }
    timing_.tracker.Stop();

    // 限制最大跟踪人数 / Cap at max persons
    if (tracked_bodies.size() > static_cast<size_t>(config_.max_persons)) {
        tracked_bodies.resize(config_.max_persons);
    }

    std::vector<DetectedObject> final_outputs;
    std::vector<uint8_t> aligned_face_buf(112 * 112 * 3);  // 对齐后人脸缓存 / Aligned face buffer (112x112 RGB)

    // ---- Step 6: 逐人体处理（ROI 裁切 → 人脸检测 → 对齐 → 特征提取 → 检索） ----
    // Process each tracked body: ROI crop → face detection → alignment → feature extraction → search

    // 将当前 body 以"仅人体"状态 fallback 输出（人脸管线中某步失败时使用）
    // Emit the current body as a person-only fallback (used when a face pipeline step fails)
    auto emit_person = [&](const DetectedObject& body_obj) {
        DetectedObject out_obj = body_obj;
        out_obj.label = CAT_PERSON;
        out_obj.bbox = body_obj.bbox;
        CoordinateMapper::NormalizeBBox(out_obj.bbox, frame_w, frame_h);
        final_outputs.push_back(out_obj);
    };

    for (const auto& body : tracked_bodies) {
        // 6a. 计算人头 ROI 区域 / Compute head ROI from body bounding box
        CropParams roi = CoordinateMapper::ComputeHeadROI(body.bbox, frame_w, frame_h, config_.head_roi);
        if (roi.w <= 0 || roi.h <= 0) {
            // ROI 无效时输出基本信息 / Output basic info when ROI is invalid
            emit_person(body);
            continue;
        }

        // 创建子图像（基于 pitch-linear 内存布局）/ Create sub-image (pitch-linear memory layout)
        const uint8_t* roi_data_ptr = frame_img.data + roi.y * frame_img.stride + roi.x * 3;
        Image head_roi{roi_data_ptr, roi.w, roi.h, 3, frame_img.stride};

        // 6b. 人脸检测预处理 / Face detection preprocessing
        timing_.face_detect.Start();
        ModelInput face_in;
        image_utils::LetterboxInfo face_letterbox;
        if (!face_detector_->Preprocess(head_roi, &face_in, &face_letterbox)) {
            timing_.face_detect.Stop();
            emit_person(body);
            continue;
        }

        // 6c. 人脸检测推理 / Face detection inference
        std::vector<ModelOutput> face_raw_outs;
        std::vector<ModelInput> face_inputs = {face_in};
        if (!face_backend_->Run(face_inputs, &face_raw_outs)) {
            timing_.face_detect.Stop();
            emit_person(body);
            continue;
        }

        // 6d. 人脸检测后处理 / Face detection postprocessing
        std::vector<DetectedObject> face_objects;
        if (!face_detector_->Postprocess(face_raw_outs, face_letterbox, &face_objects) || face_objects.empty()) {
            timing_.face_detect.Stop();
            emit_person(body);
            continue;
        }
        timing_.face_detect.Stop();

        // 6e. 选置信度最高的人脸 / Pick the highest-confidence face
        auto best_face_it = std::max_element(face_objects.begin(), face_objects.end(),
                                             [](const DetectedObject& a, const DetectedObject& b) {
                                                 return a.confidence < b.confidence;
                                             });

        DetectedObject face_obj = *best_face_it;

        // 6f. 人脸对齐（使用局部 landmarks）/ Face alignment (using local ROI landmarks)
        timing_.align.Start();
        std::array<Point, 5> local_landmarks;
        for (int p = 0; p < 5; ++p) {
            // 将 letterbox 坐标系下的 landmarks 映射回 ROI 坐标系
            // Map landmarks from letterbox coords back to ROI coords
            float kx = (face_obj.landmarks[p].x - face_letterbox.pad_x) / face_letterbox.scale;
            float ky = (face_obj.landmarks[p].y - face_letterbox.pad_y) / face_letterbox.scale;
            local_landmarks[p].x = std::max(0.0f, std::min(kx, static_cast<float>(roi.w)));
            local_landmarks[p].y = std::max(0.0f, std::min(ky, static_cast<float>(roi.h)));
        }

        // InsightFace 对齐：仿射变换到 112x112 / InsightFace alignment: affine warp to 112x112
        if (!aligner_->Align(head_roi, local_landmarks, aligned_face_buf.data())) {
            timing_.align.Stop();
            emit_person(body);
            continue;
        }
        Image aligned_face{aligned_face_buf.data(), 112, 112, 3, 112 * 3};
        timing_.align.Stop();

        // 6g. 人脸特征提取预处理 / Face recognition preprocessing
        timing_.extract.Start();
        ModelInput rec_in;
        image_utils::LetterboxInfo rec_letterbox;
        if (!face_recognizer_->Preprocess(aligned_face, &rec_in, &rec_letterbox)) {
            timing_.extract.Stop();
            emit_person(body);
            continue;
        }

        // 6h. 人脸特征提取推理 / Feature extraction inference (AdaFace)
        std::vector<ModelOutput> rec_raw_outs;
        std::vector<ModelInput> rec_inputs = {rec_in};
        if (!face_rec_backend_->Run(rec_inputs, &rec_raw_outs)) {
            timing_.extract.Stop();
            emit_person(body);
            continue;
        }

        // 6i. 特征提取后处理 / Feature extraction postprocessing (L2 normalize)
        std::vector<DetectedObject> dummy_face_list = {face_obj};
        if (!face_recognizer_->Postprocess(rec_raw_outs, rec_letterbox, &dummy_face_list)) {
            timing_.extract.Stop();
            emit_person(body);
            continue;
        }
        face_obj = dummy_face_list[0];
        timing_.extract.Stop();

        // 6j. 将人脸坐标从 ROI 坐标系映射回原图坐标系
        // Map face coordinates from ROI coords back to original frame coords
        std::vector<DetectedObject> temp_list = {face_obj};
        CoordinateMapper::MapToOriginalFrame(face_letterbox, roi, frame_w, frame_h, &temp_list);
        face_obj = temp_list[0];

        // 6k. 底库检索（余弦相似度 Top-K）/ Database search (cosine similarity Top-K)
        timing_.search.Start();
        auto search_results = face_index_.SearchTopK(face_obj.embedding.data(), 512, 5);
        timing_.search.Stop();

        // 根据检索结果设置标签和身份信息 / Set label & identity based on search results
        if (!search_results.empty() && search_results[0].similarity >= config_.recognition_threshold) {
            face_obj.label = CAT_KNOWN_FACE; // know_face / 已知人脸
            face_obj.identity_id = search_results[0].identity_id;
            face_obj.identity_name = search_results[0].identity_name;
            face_obj.similarity = search_results[0].similarity;

            // 填充候选列表（Top-K）/ Populate candidate list (Top-K)
            for (const auto& match : search_results) {
                DetectedObject::Candidate candidate;
                candidate.identity_id = match.identity_id;
                candidate.identity_name = match.identity_name;
                candidate.similarity = match.similarity;
                face_obj.candidates.push_back(candidate);
            }
        } else {
            face_obj.label = CAT_UNKNOWN_FACE; // unknown_face / 未知人脸
            face_obj.similarity = search_results.empty() ? 0.0f : search_results[0].similarity;
        }

        // 6l. 关联跟踪 ID 并归一化坐标
        // Attach track ID and normalize coordinates
        face_obj.track_id = body.track_id;
        face_obj.person_bbox = body.bbox;
        CoordinateMapper::NormalizeBBox(face_obj.person_bbox, frame_w, frame_h);
        CoordinateMapper::NormalizeBBox(face_obj.bbox, frame_w, frame_h);

        // 归一化 landmarks 到 [0, 1] / Normalize landmarks to [0, 1]
        for (auto& kp : face_obj.landmarks) {
            kp.x = std::max(0.0f, std::min(kp.x / frame_w, 1.0f));
            kp.y = std::max(0.0f, std::min(kp.y / frame_h, 1.0f));
        }

        final_outputs.push_back(face_obj);
    }

    // ---- Step 7: 构造 JSON 输出字符串 ----
    // Build JSON output string
    std::string json = "[";
    for (size_t i = 0; i < final_outputs.size(); ++i) {
        if (i > 0) json += ",";
        const auto& obj = final_outputs[i];

        json += "{";
        json += "\"category_code\":" + std::to_string(obj.label) + ",";

        // 标签文字映射 / Label text mapping
        std::string label_str;
        if (obj.label == CAT_KNOWN_FACE) label_str = "know_face";
        else if (obj.label == CAT_UNKNOWN_FACE) label_str = "unknown_face";
        else label_str = "person";

        json += "\"label\":\"" + label_str + "\",";
        json += "\"detect_confidence\":" + std::to_string(obj.confidence) + ",";
        json += "\"track_id\":" + std::to_string(obj.track_id) + ",";

        // 边界框 / Bounding box
        json += "\"bbox\":{\"x\":" + std::to_string(obj.bbox.x) +
                          ",\"y\":" + std::to_string(obj.bbox.y) +
                          ",\"w\":" + std::to_string(obj.bbox.width) +
                          ",\"h\":" + std::to_string(obj.bbox.height) + "}";

        // 人脸专属字段 / Face-specific fields
        if (obj.label == CAT_KNOWN_FACE || obj.label == CAT_UNKNOWN_FACE) {
            json += ",\"face_quality_score\":" + std::to_string(obj.confidence) + ",";
            json += "\"embedding_norm\":1.0,";

            // 5 点关键点 / 5 facial landmarks
            json += "\"landmarks\":[";
            for (size_t p = 0; p < 5; ++p) {
                if (p > 0) json += ",";
                json += "{\"x\":" + std::to_string(obj.landmarks[p].x) +
                        ",\"y\":" + std::to_string(obj.landmarks[p].y) + "}";
            }
            json += "],";

            // 关联人体框 / Associated person bounding box
            json += "\"person_bbox\":{\"x\":" + std::to_string(obj.person_bbox.x) +
                                  ",\"y\":" + std::to_string(obj.person_bbox.y) +
                                  ",\"w\":" + std::to_string(obj.person_bbox.width) +
                                  ",\"h\":" + std::to_string(obj.person_bbox.height) + "}";

            // 已知人脸才输出身份信息 / Identity fields only for known faces
            if (obj.label == CAT_KNOWN_FACE) {
                json += ",\"identity_id\":\"" + EscapeJsonString(obj.identity_id) + "\",";
                json += "\"identity_name\":\"" + EscapeJsonString(obj.identity_name) + "\",";
                json += "\"similarity\":" + std::to_string(obj.similarity) + ",";

                // 候选列表 / Candidate list
                json += "\"candidates\":[";
                for (size_t c = 0; c < obj.candidates.size(); ++c) {
                    if (c > 0) json += ",";
                    json += "{\"identity_id\":\"" + EscapeJsonString(obj.candidates[c].identity_id) + "\"" +
                            ",\"identity_name\":\"" + EscapeJsonString(obj.candidates[c].identity_name) + "\"" +
                            ",\"similarity\":" + std::to_string(obj.candidates[c].similarity) + "}";
                }
                json += "]";
            }
        }

        json += "}";
    }
    json += "]";

    // 拷贝结果到输出缓冲区 / Copy result to output buffer
    result->result_json = static_cast<char*>(std::malloc(json.size() + 1));
    std::memcpy(result->result_json, json.c_str(), json.size() + 1);
    result->result_json_len = json.size();

    timing_.total.Stop();

    // 推理总耗时（微秒）/ Total inference time in microseconds
    result->infer_time_us = static_cast<uint32_t>(timing_.total.ElapsedMs() * 1000.0);

    frame_count_++;
    // 按配置的间隔输出时序统计 / Log timing stats at configured interval
    if (config_.enable_timing_report) {
        timing_.Report(config_.log_timing_interval, frame_count_);
    }

    return 0;
}

} // namespace face_rec
