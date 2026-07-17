/**
 * @file algorithm_context.cpp
 * @brief GPU Face Recognition — Pipeline orchestrator implementation.
 */

#include "algorithm_context.h"
#include "common/logger.h"
#include "common/geometry_utils.h"
#include "postprocess/coordinate.h"
#include "preprocess/image_utils.h"

#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdlib>

namespace face_rec {

AlgorithmContext::~AlgorithmContext() {
    Destroy();
}

bool AlgorithmContext::Initialize(const char* config_json) {
    ALGO_LOGI(PIPELINE, "=== 4-Stage Initialization ===");

    // ---- Stage 1: Load configuration and logger ----
    ALGO_LOGI(PIPELINE, "Stage 1: Loading configuration...");
    config_ = Config::LoadConfig(config_json);

    // ---- Stage 2: Create inference backends (TensorRT) ----
    ALGO_LOGI(PIPELINE, "Stage 2: Creating TensorRT backends on CUDA device %d...", config_.cuda_device_id);

    person_backend_ = std::make_shared<TrtBackend>(config_.cuda_device_id, config_.trt_max_workspace_size);
    face_backend_ = std::make_shared<TrtBackend>(config_.cuda_device_id, config_.trt_max_workspace_size);
    face_rec_backend_ = std::make_shared<TrtBackend>(config_.cuda_device_id, config_.trt_max_workspace_size);

    // ---- Stage 3: Load models ----
    ALGO_LOGI(PIPELINE, "Stage 3: Loading TensorRT engines...");

    if (!person_backend_->Load(JoinPath(config_.package_dir, config_.yolo11n_engine_path))) {
        ALGO_LOGE(PIPELINE, "Failed to load YOLO11n engine");
        return false;
    }
    person_detector_ = std::make_unique<Yolo11Model>();
    person_detector_->Initialize(person_backend_);

    if (!face_backend_->Load(JoinPath(config_.package_dir, config_.retinaface_engine_path))) {
        ALGO_LOGE(PIPELINE, "Failed to load RetinaFace engine");
        return false;
    }
    face_detector_ = std::make_unique<RetinaFaceModel>();
    face_detector_->Initialize(face_backend_);

    if (!face_rec_backend_->Load(JoinPath(config_.package_dir, config_.adaface_ir50_engine_path))) {
        ALGO_LOGE(PIPELINE, "Failed to load AdaFace engine");
        return false;
    }
    face_recognizer_ = std::make_unique<AdaFaceModel>();
    face_recognizer_->Initialize(face_rec_backend_);

    // ---- Stage 4: Create pipeline components ----
    ALGO_LOGI(PIPELINE, "Stage 4: Creating pipeline components...");

    aligner_ = std::make_unique<FaceAligner>();
    tracker_ = std::make_unique<ByteTracker>(
        config_.tracker_iou_thres, config_.tracker_max_lost_frames);
    thread_pool_ = std::make_unique<ThreadPool>(2);  // 1 GPU inference + 1 FAISS matching

    // Pre-allocate buffers
    aligned_face_buffer_.resize(112 * 112 * 3);

    initialized_ = true;
    ALGO_LOGI(PIPELINE, "=== Initialization Complete ===");
    ALGO_LOGI(PIPELINE, "  YOLO11n: %s", config_.yolo11n_engine_path.c_str());
    ALGO_LOGI(PIPELINE, "  RetinaFace: %s", config_.retinaface_engine_path.c_str());
    ALGO_LOGI(PIPELINE, "  AdaFace: %s", config_.adaface_ir50_engine_path.c_str());

    return true;
}

int AlgorithmContext::Infer(const hw_buffer_desc_t* input, const char* context_json,
                            infer_result_t* result) {
    if (!initialized_ || !input || !result) return -1;
    if (!input->data || input->width == 0 || input->height == 0) return -2;

    std::lock_guard<std::mutex> lock(infer_mutex_);
    timing_.Reset();
    timing_.total.Start();

    frame_count_++;

    // Determine input format and get BGR data
    const uint8_t* bgr_data = nullptr;
    int width = input->width;
    int height = input->height;

    if (input->pixel_format == pixel_format::NV12) {
        // NV12 → BGR conversion
        size_t bgr_size = width * height * 3;
        if (bgr_buffer_.size() < bgr_size) {
            bgr_buffer_.resize(bgr_size);
        }
        Nv12ToBgr(static_cast<const uint8_t*>(input->data), width, height,
                  input->stride, bgr_buffer_.data());
        bgr_data = bgr_buffer_.data();
    } else {
        // Assume BGR24
        bgr_data = static_cast<const uint8_t*>(input->data);
    }

    int ret = InferFrame(bgr_data, width, height, context_json, result);

    timing_.total.Stop();
    timing_.Report(config_.log_timing_interval, frame_count_);

    return ret;
}

int AlgorithmContext::InferFrame(const uint8_t* bgr_data, int width, int height,
                                  const char* context_json, infer_result_t* result) {
    // ---- Step 1: Person Detection (YOLO11n) ----
    timing_.person_detect.Start();
    Image frame_img;
    frame_img.data = bgr_data;
    frame_img.width = width;
    frame_img.height = height;
    frame_img.channels = 3;
    frame_img.stride = width;

    std::vector<Image> batch = {frame_img};
    auto person_results = person_detector_->DetectBatch(
        batch, config_.person_conf_thres, config_.nms_iou_thres);
    timing_.person_detect.Stop();

    // ---- Step 2: Tracking (ByteTrack) ----
    timing_.tracker.Start();
    std::vector<DetectedObject> tracked_persons;
    if (config_.enable_tracker && !person_results.empty()) {
        tracked_persons = tracker_->Update(person_results[0]);
    } else if (!person_results.empty()) {
        tracked_persons = person_results[0];
    }
    timing_.tracker.Stop();

    // ---- Step 3: Face Detection (RetinaFace) ----
    timing_.face_detect.Start();
    auto face_results = face_detector_->DetectBatch(
        batch, config_.face_conf_thres, config_.nms_iou_thres);
    timing_.face_detect.Stop();

    // ---- Step 4: IoU Association (Person ↔ Face) ----
    std::vector<DetectedObject> faces;
    if (!face_results.empty()) {
        faces = face_results[0];
    }

    // Associate faces with tracked persons via IoU
    for (auto& face : faces) {
        float best_iou = 0.1f;
        int best_person_idx = -1;
        for (size_t p = 0; p < tracked_persons.size(); ++p) {
            float iou = IoU(face.bbox, tracked_persons[p].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_person_idx = static_cast<int>(p);
            }
        }
        if (best_person_idx >= 0) {
            face.track_id = tracked_persons[best_person_idx].track_id;
            face.person_bbox = tracked_persons[best_person_idx].bbox;
        }
    }

    // ---- Step 5: Face Alignment ----
    timing_.align.Start();
    std::vector<Image> face_images;
    std::vector<std::array<Point, 5>> landmarks_batch;

    for (const auto& face : faces) {
        // Clamp face bbox to image bounds
        Rect clamped = ClampRect(face.bbox, width, height);
        if (!IsValidRect(clamped) || clamped.width < 20 || clamped.height < 20) continue;

        // Create a sub-image view for the face region
        CropParams crop;
        crop.x = static_cast<int>(clamped.x);
        crop.y = static_cast<int>(clamped.y);
        crop.w = static_cast<int>(clamped.width);
        crop.h = static_cast<int>(clamped.height);

        // Crop and store
        std::vector<uint8_t> cropped(crop.w * crop.h * 3);
        CropImage(bgr_data, width, height, width, cropped.data(), crop);

        Image cropped_img;
        cropped_img.data = cropped.data();
        cropped_img.width = crop.w;
        cropped_img.height = crop.h;
        cropped_img.channels = 3;
        cropped_img.stride = crop.w;

        face_images.push_back(cropped_img);

        // Adjust landmarks relative to crop
        std::array<Point, 5> adj_landmarks;
        for (int l = 0; l < 5; ++l) {
            adj_landmarks[l].x = face.landmarks[l].x - clamped.x;
            adj_landmarks[l].y = face.landmarks[l].y - clamped.y;
        }
        landmarks_batch.push_back(adj_landmarks);
    }
    timing_.align.Stop();

    // ---- Step 6: Feature Extraction (AdaFace) ----
    timing_.extract.Start();
    if (!face_images.empty()) {
        auto aligned_faces = aligner_->AlignBatch(face_images, landmarks_batch);
        auto embeddings = face_recognizer_->ExtractBatch(aligned_faces);

        for (size_t i = 0; i < embeddings.size() && i < faces.size(); ++i) {
            faces[i].embedding = embeddings[i];
        }
    }
    timing_.extract.Stop();

    // ---- Step 7: Gallery Matching ----
    timing_.search.Start();
    for (auto& face : faces) {
        if (face.embedding.empty()) {
            face.label = CAT_UNKNOWN_FACE;
            continue;
        }

        std::string matched_id, matched_name;
        float similarity = 0.0f;
        std::vector<DetectedObject::Candidate> candidates;

        bool matched = face_index_.Search(face.embedding, config_.recognition_threshold,
                                           matched_id, matched_name, similarity, candidates);

        if (matched) {
            face.identity_id = matched_id;
            face.identity_name = matched_name;
            face.similarity = similarity;
            face.candidates = candidates;
            face.label = CAT_KNOWN_FACE;
        } else {
            face.label = CAT_UNKNOWN_FACE;
            face.candidates = candidates;
        }

        // Update short-term store
        if (face.track_id >= 0) {
            face_index_.AddShortTerm(face.track_id, face.embedding,
                                      face.identity_id, face.identity_name);
        }
    }
    timing_.search.Stop();

    // ---- Step 8: Build Result JSON ----
    std::string json = BuildResultJson(faces, width, height);

    result->result_json = static_cast<char*>(std::malloc(json.size() + 1));
    if (!result->result_json) return -3;
    std::memcpy(result->result_json, json.c_str(), json.size() + 1);
    result->result_json_len = json.size();
    result->infer_time_us = static_cast<uint32_t>(timing_.total.ElapsedMs() * 1000);

    return 0;
}

std::string AlgorithmContext::BuildResultJson(const std::vector<DetectedObject>& faces,
                                               int img_w, int img_h) {
    std::ostringstream ss;
    ss << "[";

    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& face = faces[i];
        if (i > 0) ss << ",";

        Rect norm_bbox = NormalizeBbox(face.bbox, img_w, img_h);

        ss << "{";
        ss << "\"category_code\":" << face.label << ",";
        ss << "\"detect_confidence\":" << face.confidence << ",";
        ss << "\"bbox\":[" << norm_bbox.x << "," << norm_bbox.y << ","
           << norm_bbox.width << "," << norm_bbox.height << "],";
        ss << "\"track_id\":" << face.track_id << ",";

        // Landmarks
        ss << "\"face_landmarks\":[";
        auto norm_lm = NormalizeLandmarks(face.landmarks, img_w, img_h);
        for (int l = 0; l < 5; ++l) {
            if (l > 0) ss << ",";
            ss << "[" << norm_lm[l].x << "," << norm_lm[l].y << "]";
        }
        ss << "],";

        ss << "\"identity_id\":\"" << face.identity_id << "\",";
        ss << "\"identity_name\":\"" << face.identity_name << "\",";
        ss << "\"similarity\":" << face.similarity << ",";

        // Candidates
        ss << "\"candidates\":[";
        for (size_t c = 0; c < face.candidates.size(); ++c) {
            if (c > 0) ss << ",";
            ss << "{\"identity_id\":\"" << face.candidates[c].identity_id
               << "\",\"identity_name\":\"" << face.candidates[c].identity_name
               << "\",\"similarity\":" << face.candidates[c].similarity << "}";
        }
        ss << "]";

        ss << "}";
    }

    ss << "]";
    return ss.str();
}

int AlgorithmContext::UpdateFaceLibrary(const char* face_library_json) {
    if (!face_library_json) return -1;

    // Parse JSON array of identity entries
    // Simplified JSON parsing for gallery update
    std::vector<IdentityEntry> entries;

    // In production, this would use nlohmann_json or similar
    // For now, expect a minimal format: [{"id":"...","name":"...","embedding":[...]}]
    // This is a placeholder — the actual JSON parsing would be implemented
    // with the project's JSON library.

    ALGO_LOGI(FACE_INDEX, "Updating face library (JSON len=%zu)", std::strlen(face_library_json));

    face_index_.RebuildGallery(entries);
    return 0;
}

void AlgorithmContext::Destroy() {
    if (!initialized_) return;

    ALGO_LOGI(PIPELINE, "Destroying algorithm context...");

    face_index_.ClearShortTerm();
    tracker_.reset();
    aligner_.reset();
    face_recognizer_.reset();
    face_detector_.reset();
    person_detector_.reset();

    face_rec_backend_.reset();
    face_backend_.reset();
    person_backend_.reset();

    initialized_ = false;
    ALGO_LOGI(PIPELINE, "Algorithm context destroyed");
}

} // namespace face_rec
