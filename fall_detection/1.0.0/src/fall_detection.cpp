#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

#include "algo/abi_contract.h"

namespace {

static void AlgoLog(const char* level, const char* fmt, ...) {
    std::fprintf(stderr, "[fall_detection][%s] ", level);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
}

#define ALGO_LOG_ERROR(...) AlgoLog("ERROR", __VA_ARGS__)
#define ALGO_LOG_WARN(...)  AlgoLog("WARN", __VA_ARGS__)
#define ALGO_LOG_INFO(...)  AlgoLog("INFO", __VA_ARGS__)

constexpr int kCategoryFall = 21001;
constexpr int kCategorySuspectedFall = 21002;
constexpr int kCategoryNormalPerson = 21003;
constexpr int kKeypointCount = 17;
constexpr int kModelInputSize = 640;
constexpr float kPi = 3.14159265358979323846f;

// COCO 17 点下标
constexpr int KP_NOSE = 0;
constexpr int KP_LEFT_SHOULDER = 5;
constexpr int KP_RIGHT_SHOULDER = 6;
constexpr int KP_LEFT_HIP = 11;
constexpr int KP_RIGHT_HIP = 12;
constexpr int KP_LEFT_ANKLE = 15;
constexpr int KP_RIGHT_ANKLE = 16;

enum class AlgoError : int {
    SUCCESS = 0,
    INVALID_HANDLE = -1,
    INVALID_INPUT = -2,
    INVALID_OUTPUT = -3,
    INIT_FAILED = -4,
    INFER_FAILED = -5,
    UNSUPPORTED_INPUT = -6,
};

struct Config {
    std::string package_dir;
    std::string model_path = "models/yolov8n-pose.onnx";
    int input_size = kModelInputSize;
    float conf_thres = 0.45f;
    float iou_thres = 0.45f;
    float keypoint_thres = 0.35f;
    float fall_aspect_ratio = 1.25f;
    float torso_tilt_degree = 60.0f;
    int min_pose_points = 5;
    float suspected_score = 0.45f;
    float fall_score = 0.65f;
    int max_detections = 20;
};

struct LetterboxInfo {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
};

struct PoseDetection {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float confidence = 0.0f;
    std::vector<Keypoint> keypoints;
};

struct FallAnalysis {
    int category_code = kCategoryNormalPerson;
    std::string label = "normal_person";
    float fall_score = 0.0f;
    float aspect_ratio = 0.0f;
    float torso_tilt_degree = -1.0f;
    int valid_keypoints = 0;
};

struct AlgorithmContext {
    Config config;
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;
    std::vector<std::string> input_names_storage;
    std::vector<std::string> output_names_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
};

static std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n\"");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n\"");
    return value.substr(begin, end - begin + 1);
}

static bool ExtractString(const std::string& json, const std::string& key, std::string* out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const auto quote_begin = json.find('"', colon_pos + 1);
    if (quote_begin == std::string::npos) {
        return false;
    }
    const auto quote_end = json.find('"', quote_begin + 1);
    if (quote_end == std::string::npos) {
        return false;
    }
    *out = json.substr(quote_begin + 1, quote_end - quote_begin - 1);
    return true;
}

static bool ExtractNumber(const std::string& json, const std::string& key, double* out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const auto value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return false;
    }
    const auto value_end = json.find_first_of(",}\r\n", value_begin);
    const std::string raw = json.substr(value_begin, value_end == std::string::npos ? std::string::npos : value_end - value_begin);
    char* end_ptr = nullptr;
    const double parsed = std::strtod(Trim(raw).c_str(), &end_ptr);
    if (end_ptr == nullptr) {
        return false;
    }
    *out = parsed;
    return true;
}

static std::string JoinPath(const std::string& base, const std::string& path) {
    if (path.empty()) {
        return base;
    }
    if (!path.empty() && path[0] == '/') {
        return path;
    }
    if (base.empty()) {
        return path;
    }
    if (base.back() == '/') {
        return base + path;
    }
    return base + "/" + path;
}

static std::string DirName(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

static std::string ResolveSharedLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ResolveSharedLibraryDir), &info) != 0 && info.dli_fname) {
        return DirName(info.dli_fname);
    }
    return ".";
}

static Config ParseConfig(const char* config_json) {
    Config cfg;
    const std::string json(config_json ? config_json : "");

    ExtractString(json, "package_dir", &cfg.package_dir);
    ExtractString(json, "model_path", &cfg.model_path);

    double value = 0.0;
    if (ExtractNumber(json, "input_size", &value)) cfg.input_size = static_cast<int>(value);
    if (ExtractNumber(json, "conf_thres", &value)) cfg.conf_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "iou_thres", &value)) cfg.iou_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "keypoint_thres", &value)) cfg.keypoint_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "fall_aspect_ratio", &value)) cfg.fall_aspect_ratio = std::clamp(static_cast<float>(value), 0.5f, 5.0f);
    if (ExtractNumber(json, "torso_tilt_degree", &value)) cfg.torso_tilt_degree = std::clamp(static_cast<float>(value), 0.0f, 90.0f);
    if (ExtractNumber(json, "min_pose_points", &value)) cfg.min_pose_points = std::clamp(static_cast<int>(value), 3, 17);
    if (ExtractNumber(json, "suspected_score", &value)) cfg.suspected_score = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "fall_score", &value)) cfg.fall_score = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "max_detections", &value)) cfg.max_detections = std::clamp(static_cast<int>(value), 1, 100);

    if (cfg.suspected_score > cfg.fall_score) {
        cfg.suspected_score = cfg.fall_score;
    }
    if (cfg.package_dir.empty()) {
        cfg.package_dir = ResolveSharedLibraryDir();
    }
    if (cfg.input_size != kModelInputSize) {
        ALGO_LOG_WARN("static ONNX model only supports input_size=%d, got %d; using %d",
                      kModelInputSize,
                      cfg.input_size,
                      kModelInputSize);
        cfg.input_size = kModelInputSize;
    }
    return cfg;
}

static bool FileExists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

static cv::Mat ConvertInputToMat(const hw_buffer_desc_t* input);

static cv::Mat Letterbox(const cv::Mat& image, int input_size, LetterboxInfo* info) {
    const int width = image.cols;
    const int height = image.rows;
    const float scale = std::min(static_cast<float>(input_size) / static_cast<float>(width),
                                 static_cast<float>(input_size) / static_cast<float>(height));
    const int new_width = static_cast<int>(std::round(width * scale));
    const int new_height = static_cast<int>(std::round(height * scale));
    const int pad_x = (input_size - new_width) / 2;
    const int pad_y = (input_size - new_height) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));

    cv::Mat output(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(output(cv::Rect(pad_x, pad_y, new_width, new_height)));

    info->scale = scale;
    info->pad_x = static_cast<float>(pad_x);
    info->pad_y = static_cast<float>(pad_y);
    return output;
}

static float IoU(const PoseDetection& a, const PoseDetection& b) {
    const float ax1 = a.x;
    const float ay1 = a.y;
    const float ax2 = a.x + a.w;
    const float ay2 = a.y + a.h;
    const float bx1 = b.x;
    const float by1 = b.y;
    const float bx2 = b.x + b.w;
    const float by2 = b.y + b.h;

    const float inter_x1 = std::max(ax1, bx1);
    const float inter_y1 = std::max(ay1, by1);
    const float inter_x2 = std::min(ax2, bx2);
    const float inter_y2 = std::min(ay2, by2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    const float union_area = a.w * a.h + b.w * b.h - inter_area;
    return union_area <= 0.0f ? 0.0f : inter_area / union_area;
}

static std::vector<PoseDetection> ApplyNms(std::vector<PoseDetection> detections, float iou_thres, int max_detections) {
    std::sort(detections.begin(), detections.end(), [](const PoseDetection& lhs, const PoseDetection& rhs) {
        return lhs.confidence > rhs.confidence;
    });

    std::vector<PoseDetection> kept;
    std::vector<bool> removed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        kept.push_back(detections[i]);
        if (static_cast<int>(kept.size()) >= max_detections) {
            break;
        }
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!removed[j] && IoU(detections[i], detections[j]) > iou_thres) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

static float RestoreCoord(float value, float pad, float scale, float max_value) {
    const float restored = (value - pad) / scale;
    return std::clamp(restored, 0.0f, max_value);
}

static bool KeypointValid(const PoseDetection& det, int index, float thres) {
    return index >= 0 && index < static_cast<int>(det.keypoints.size()) && det.keypoints[index].score >= thres;
}

static bool Midpoint(const PoseDetection& det, int left_idx, int right_idx, float thres, cv::Point2f* out) {
    const bool left_valid = KeypointValid(det, left_idx, thres);
    const bool right_valid = KeypointValid(det, right_idx, thres);
    if (!left_valid && !right_valid) {
        return false;
    }
    if (left_valid && right_valid) {
        out->x = (det.keypoints[left_idx].x + det.keypoints[right_idx].x) * 0.5f;
        out->y = (det.keypoints[left_idx].y + det.keypoints[right_idx].y) * 0.5f;
        return true;
    }
    const Keypoint& kp = left_valid ? det.keypoints[left_idx] : det.keypoints[right_idx];
    out->x = kp.x;
    out->y = kp.y;
    return true;
}

static FallAnalysis AnalyzeFall(const PoseDetection& det, const Config& cfg) {
    FallAnalysis analysis;
    analysis.aspect_ratio = det.h <= 1.0f ? 0.0f : det.w / det.h;

    for (const auto& kp : det.keypoints) {
        if (kp.score >= cfg.keypoint_thres) {
            analysis.valid_keypoints++;
        }
    }

    float score = 0.0f;
    if (analysis.aspect_ratio >= cfg.fall_aspect_ratio) {
        score += 0.40f;
    } else if (analysis.aspect_ratio >= cfg.fall_aspect_ratio * 0.85f) {
        score += 0.25f;
    }

    cv::Point2f shoulder_mid;
    cv::Point2f hip_mid;
    if (Midpoint(det, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, cfg.keypoint_thres, &shoulder_mid) &&
        Midpoint(det, KP_LEFT_HIP, KP_RIGHT_HIP, cfg.keypoint_thres, &hip_mid)) {
        const float dx = std::abs(hip_mid.x - shoulder_mid.x);
        const float dy = std::abs(hip_mid.y - shoulder_mid.y);
        analysis.torso_tilt_degree = std::atan2(dx, std::max(dy, 1.0f)) * 180.0f / kPi;
        if (analysis.torso_tilt_degree >= cfg.torso_tilt_degree) {
            score += 0.40f;
        } else if (analysis.torso_tilt_degree >= cfg.torso_tilt_degree * 0.75f) {
            score += 0.25f;
        }

        // 肩髋垂直距离很小，通常表示人体横向倒地。
        if (dy <= det.h * 0.35f) {
            score += 0.10f;
        }
    } else if (analysis.valid_keypoints < cfg.min_pose_points) {
        score += 0.10f;
    }

    // 鼻子与脚踝在垂直方向接近时增强横倒判断。
    cv::Point2f ankle_mid;
    if (KeypointValid(det, KP_NOSE, cfg.keypoint_thres) &&
        Midpoint(det, KP_LEFT_ANKLE, KP_RIGHT_ANKLE, cfg.keypoint_thres, &ankle_mid)) {
        const float vertical_span = std::abs(ankle_mid.y - det.keypoints[KP_NOSE].y);
        if (vertical_span <= det.h * 0.65f) {
            score += 0.10f;
        }
    }

    // 关键点不足时降低确定性，但保留 bbox 横向导致的疑似告警能力。
    if (analysis.valid_keypoints < cfg.min_pose_points) {
        score *= 0.75f;
    }

    analysis.fall_score = std::clamp(score, 0.0f, 1.0f);
    if (analysis.fall_score >= cfg.fall_score) {
        analysis.category_code = kCategoryFall;
        analysis.label = "fall";
    } else if (analysis.fall_score >= cfg.suspected_score) {
        analysis.category_code = kCategorySuspectedFall;
        analysis.label = "suspected_fall";
    } else {
        analysis.category_code = kCategoryNormalPerson;
        analysis.label = "normal_person";
    }
    return analysis;
}

static std::string EscapeJson(const std::string& value) {
    std::ostringstream oss;
    for (char ch : value) {
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default: oss << ch; break;
        }
    }
    return oss.str();
}

static std::string BuildResultJson(const std::vector<PoseDetection>& detections, const std::vector<FallAnalysis>& analyses) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(4);
    oss << "[";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        const auto& analysis = analyses[i];
        if (i > 0) {
            oss << ",";
        }
        oss << "{"
            << "\"category_code\":" << analysis.category_code << ","
            << "\"label\":\"" << EscapeJson(analysis.label) << "\","
            << "\"detect_confidence\":" << det.confidence << ","
            << "\"fall_score\":" << analysis.fall_score << ","
            << "\"aspect_ratio\":" << analysis.aspect_ratio << ","
            << "\"torso_tilt_degree\":" << analysis.torso_tilt_degree << ","
            << "\"valid_keypoints\":" << analysis.valid_keypoints << ","
            << "\"bbox\":{"
            << "\"x\":" << det.x << ","
            << "\"y\":" << det.y << ","
            << "\"w\":" << det.w << ","
            << "\"h\":" << det.h << "},"
            << "\"keypoints\":[";
        for (size_t k = 0; k < det.keypoints.size(); ++k) {
            if (k > 0) {
                oss << ",";
            }
            oss << "{\"x\":" << det.keypoints[k].x
                << ",\"y\":" << det.keypoints[k].y
                << ",\"score\":" << det.keypoints[k].score << "}";
        }
        oss << "]}";
    }
    oss << "]";
    return oss.str();
}

static bool CopyJsonToResult(const std::string& json, uint32_t infer_time_us, infer_result_t* result) {
    if (!result) {
        return false;
    }
    result->result_json = static_cast<char*>(std::malloc(json.size() + 1));
    if (!result->result_json) {
        return false;
    }
    std::memcpy(result->result_json, json.c_str(), json.size() + 1);
    result->result_json_len = json.size();
    result->infer_time_us = infer_time_us;
    std::memset(result->reserved, 0, sizeof(result->reserved));
    return true;
}

static std::vector<PoseDetection> DecodeOutput(const float* data,
                                                const std::vector<int64_t>& shape,
                                                int image_width,
                                                int image_height,
                                                const LetterboxInfo& letterbox,
                                                const Config& cfg) {
    std::vector<PoseDetection> detections;
    if (!data || shape.size() < 3) {
        return detections;
    }

    // 支持 [1, 56, 8400] 与 [1, 8400, 56] 两种导出形态。
    int64_t rows = 0;
    int64_t dims = 0;
    bool transposed = false;
    if (shape[1] == 56 || shape[1] == 57) {
        dims = shape[1];
        rows = shape[2];
        transposed = true;
    } else {
        rows = shape[1];
        dims = shape[2];
        transposed = false;
    }

    if (dims < 5 + kKeypointCount * 3) {
        ALGO_LOG_ERROR("YOLOv8-pose output dims invalid: %lld", static_cast<long long>(dims));
        return detections;
    }

    auto at = [&](int64_t row, int64_t dim) -> float {
        if (transposed) {
            return data[dim * rows + row];
        }
        return data[row * dims + dim];
    };

    for (int64_t i = 0; i < rows; ++i) {
        const float confidence = at(i, 4);
        if (confidence < cfg.conf_thres) {
            continue;
        }

        const float cx = at(i, 0);
        const float cy = at(i, 1);
        const float w = at(i, 2);
        const float h = at(i, 3);
        const float x1_model = cx - w * 0.5f;
        const float y1_model = cy - h * 0.5f;
        const float x2_model = cx + w * 0.5f;
        const float y2_model = cy + h * 0.5f;

        const float x1 = RestoreCoord(x1_model, letterbox.pad_x, letterbox.scale, static_cast<float>(image_width - 1));
        const float y1 = RestoreCoord(y1_model, letterbox.pad_y, letterbox.scale, static_cast<float>(image_height - 1));
        const float x2 = RestoreCoord(x2_model, letterbox.pad_x, letterbox.scale, static_cast<float>(image_width - 1));
        const float y2 = RestoreCoord(y2_model, letterbox.pad_y, letterbox.scale, static_cast<float>(image_height - 1));

        PoseDetection det;
        det.x = std::min(x1, x2);
        det.y = std::min(y1, y2);
        det.w = std::max(0.0f, std::abs(x2 - x1));
        det.h = std::max(0.0f, std::abs(y2 - y1));
        det.confidence = confidence;
        if (det.w < 2.0f || det.h < 2.0f) {
            continue;
        }

        det.keypoints.resize(kKeypointCount);
        for (int k = 0; k < kKeypointCount; ++k) {
            const int base = 5 + k * 3;
            det.keypoints[k].x = RestoreCoord(at(i, base), letterbox.pad_x, letterbox.scale, static_cast<float>(image_width - 1));
            det.keypoints[k].y = RestoreCoord(at(i, base + 1), letterbox.pad_y, letterbox.scale, static_cast<float>(image_height - 1));
            det.keypoints[k].score = at(i, base + 2);
        }
        detections.push_back(std::move(det));
    }

    return ApplyNms(std::move(detections), cfg.iou_thres, cfg.max_detections);
}

static std::vector<PoseDetection> RunInference(AlgorithmContext* ctx, const cv::Mat& image) {
    LetterboxInfo letterbox;
    cv::Mat input_image = Letterbox(image, ctx->config.input_size, &letterbox);

    cv::Mat blob = cv::dnn::blobFromImage(input_image, 1.0 / 255.0, cv::Size(ctx->config.input_size, ctx->config.input_size), cv::Scalar(), true, false, CV_32F);
    std::array<int64_t, 4> input_shape = {1, 3, ctx->config.input_size, ctx->config.input_size};
    const size_t input_tensor_size = static_cast<size_t>(ctx->config.input_size) * static_cast<size_t>(ctx->config.input_size) * 3;

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              reinterpret_cast<float*>(blob.data),
                                                              input_tensor_size,
                                                              input_shape.data(),
                                                              input_shape.size());

    auto output_tensors = ctx->session->Run(Ort::RunOptions{nullptr},
                                            ctx->input_names.data(),
                                            &input_tensor,
                                            1,
                                            ctx->output_names.data(),
                                            ctx->output_names.size());
    if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
        throw std::runtime_error("invalid output tensor");
    }

    const float* output_data = output_tensors[0].GetTensorData<float>();
    const std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    return DecodeOutput(output_data, output_shape, image.cols, image.rows, letterbox, ctx->config);
}

static bool InitializeSession(AlgorithmContext* ctx) {
    const std::string model_path = JoinPath(ctx->config.package_dir, ctx->config.model_path);
    if (!FileExists(model_path)) {
        ALGO_LOG_ERROR("model not found: %s", model_path.c_str());
        return false;
    }

    ctx->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "fall_detection");
    ctx->session_options.SetIntraOpNumThreads(1);
    ctx->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    ctx->session = std::make_unique<Ort::Session>(*ctx->env, model_path.c_str(), ctx->session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = ctx->session->GetInputCount();
    const size_t output_count = ctx->session->GetOutputCount();
    if (input_count == 0 || output_count == 0) {
        ALGO_LOG_ERROR("model has no input or output");
        return false;
    }

    for (size_t i = 0; i < input_count; ++i) {
        auto name = ctx->session->GetInputNameAllocated(i, allocator);
        ctx->input_names_storage.emplace_back(name.get());
    }
    for (size_t i = 0; i < output_count; ++i) {
        auto name = ctx->session->GetOutputNameAllocated(i, allocator);
        ctx->output_names_storage.emplace_back(name.get());
    }
    for (const auto& name : ctx->input_names_storage) {
        ctx->input_names.push_back(name.c_str());
    }
    for (const auto& name : ctx->output_names_storage) {
        ctx->output_names.push_back(name.c_str());
    }

    ctx->input_shape = ctx->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    ALGO_LOG_INFO("loaded YOLOv8-pose model: %s", model_path.c_str());
    return true;
}

} // namespace

namespace {

constexpr uint32_t FourCC(char a, char b, char c, char d) {
    return static_cast<uint32_t>(a) |
           (static_cast<uint32_t>(b) << 8) |
           (static_cast<uint32_t>(c) << 16) |
           (static_cast<uint32_t>(d) << 24);
}

constexpr uint32_t kPixelFormatRGB24 = FourCC('R', 'G', 'B', '3');
constexpr uint32_t kPixelFormatGray = FourCC('G', 'R', 'E', 'Y');
constexpr uint32_t kPixelFormatNV12 = FourCC('N', 'V', '1', '2');
constexpr uint32_t kPixelFormatNV21 = FourCC('N', 'V', '2', '1');

static cv::Mat ConvertMappedInputToMat(const unsigned char* data, const hw_buffer_desc_t* input) {
    const size_t pixel_count = static_cast<size_t>(input->width) * static_cast<size_t>(input->height);

    if ((input->pixel_format == kPixelFormatNV12 || input->pixel_format == kPixelFormatNV21 ||
         (input->pixel_format == 0 && input->size >= pixel_count * 3 / 2 && input->size < pixel_count * 3)) &&
        input->size >= pixel_count * 3 / 2) {
        cv::Mat yuv(static_cast<int>(input->height + input->height / 2),
                    static_cast<int>(input->width),
                    CV_8UC1,
                    const_cast<unsigned char*>(data));
        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, input->pixel_format == kPixelFormatNV21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12);
        return bgr;
    }

    if (input->size >= pixel_count * 3) {
        cv::Mat image(static_cast<int>(input->height),
                      static_cast<int>(input->width),
                      CV_8UC3,
                      const_cast<unsigned char*>(data));
        if (input->pixel_format == kPixelFormatRGB24) {
            cv::Mat bgr;
            cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
            return bgr;
        }
        return image.clone();
    }

    if (input->size >= pixel_count) {
        cv::Mat gray(static_cast<int>(input->height),
                     static_cast<int>(input->width),
                     CV_8UC1,
                     const_cast<unsigned char*>(data));
        if (input->pixel_format == kPixelFormatGray || input->pixel_format == 0) {
            cv::Mat bgr;
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            return bgr;
        }
    }

    ALGO_LOG_ERROR("unsupported input buffer size=%zu format=%u", input->size, input->pixel_format);
    return cv::Mat();
}

static cv::Mat ConvertInputToMat(const hw_buffer_desc_t* input) {
    if (!input || input->width == 0 || input->height == 0 || input->size == 0) {
        return cv::Mat();
    }

    const int fd = input->dma_buf_fd >= 0 ? input->dma_buf_fd : input->dma_fd;
    if (fd < 0) {
        ALGO_LOG_ERROR("input has no dma fd");
        return cv::Mat();
    }

    void* mapped = mmap(nullptr, input->size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        ALGO_LOG_ERROR("mmap failed for fd=%d", fd);
        return cv::Mat();
    }

    cv::Mat image = ConvertMappedInputToMat(static_cast<const unsigned char*>(mapped), input);
    munmap(mapped, input->size);
    return image;
}

} // namespace

extern "C" {

algo_handle_t detector_init(const char* config_json) {
    if (!config_json) {
        ALGO_LOG_ERROR("detector_init: config_json is null");
        return nullptr;
    }

    try {
        auto ctx = std::make_unique<AlgorithmContext>();
        ctx->config = ParseConfig(config_json);
        if (ctx->config.package_dir.empty()) {
            ALGO_LOG_ERROR("detector_init: package_dir is required");
            return nullptr;
        }
        if (!InitializeSession(ctx.get())) {
            return nullptr;
        }
        return reinterpret_cast<algo_handle_t>(ctx.release());
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_init exception: %s", e.what());
        return nullptr;
    } catch (...) {
        ALGO_LOG_ERROR("detector_init unknown exception");
        return nullptr;
    }
}

int detector_infer(algo_handle_t handle, const hw_buffer_desc_t* input, const char* /*context_json*/, infer_result_t* result) {
    const auto start = std::chrono::steady_clock::now();
    if (!handle) {
        return static_cast<int>(AlgoError::INVALID_HANDLE);
    }
    if (!input) {
        return static_cast<int>(AlgoError::INVALID_INPUT);
    }
    if (!result) {
        return static_cast<int>(AlgoError::INVALID_OUTPUT);
    }
    result->result_json = nullptr;
    result->result_json_len = 0;
    result->infer_time_us = 0;
    std::memset(result->reserved, 0, sizeof(result->reserved));

    try {
        auto* ctx = reinterpret_cast<AlgorithmContext*>(handle);
        cv::Mat image = ConvertInputToMat(input);
        if (image.empty()) {
            return static_cast<int>(AlgoError::UNSUPPORTED_INPUT);
        }

        std::vector<PoseDetection> detections = RunInference(ctx, image);
        std::vector<FallAnalysis> analyses;
        analyses.reserve(detections.size());
        for (const auto& det : detections) {
            analyses.push_back(AnalyzeFall(det, ctx->config));
        }

        const auto end = std::chrono::steady_clock::now();
        const auto infer_time_us = static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        const std::string json = BuildResultJson(detections, analyses);
        if (!CopyJsonToResult(json, infer_time_us, result)) {
            return static_cast<int>(AlgoError::INVALID_OUTPUT);
        }
        return static_cast<int>(AlgoError::SUCCESS);
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_infer exception: %s", e.what());
        return static_cast<int>(AlgoError::INFER_FAILED);
    } catch (...) {
        ALGO_LOG_ERROR("detector_infer unknown exception");
        return static_cast<int>(AlgoError::INFER_FAILED);
    }
}

void detector_free_result(infer_result_t* result) {
    if (!result) {
        return;
    }
    if (result->result_json) {
        std::free(result->result_json);
    }
    result->result_json = nullptr;
    result->result_json_len = 0;
    result->infer_time_us = 0;
    std::memset(result->reserved, 0, sizeof(result->reserved));
}

void detector_destroy(algo_handle_t handle) {
    try {
        auto* ctx = reinterpret_cast<AlgorithmContext*>(handle);
        delete ctx;
    } catch (...) {
        // C ABI 边界禁止抛出异常。
    }
}

const char* detector_version(void) {
    return "1.0.0";
}

const char* detector_name(void) {
    return "fall_detection";
}

int detector_self_test(void) {
    try {
        const std::string config = R"({"model_path":"models/yolov8n-pose.onnx","input_size":640})";
        algo_handle_t handle = detector_init(config.c_str());
        if (!handle) {
            ALGO_LOG_ERROR("self_test: detector_init failed");
            return static_cast<int>(AlgoError::INIT_FAILED);
        }

        auto* ctx = reinterpret_cast<AlgorithmContext*>(handle);
        cv::Mat image = cv::imread(JoinPath(ctx->config.package_dir, "testimage.jpg"), cv::IMREAD_COLOR);
        if (image.empty()) {
            ALGO_LOG_ERROR("self_test: testimage.jpg not found or invalid");
            detector_destroy(handle);
            return static_cast<int>(AlgoError::INVALID_INPUT);
        }

        std::vector<PoseDetection> detections = RunInference(ctx, image);
        std::vector<FallAnalysis> analyses;
        analyses.reserve(detections.size());
        for (const auto& det : detections) {
            analyses.push_back(AnalyzeFall(det, ctx->config));
        }
        (void)BuildResultJson(detections, analyses);
        detector_destroy(handle);
        return static_cast<int>(AlgoError::SUCCESS);
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_self_test exception: %s", e.what());
        return static_cast<int>(AlgoError::INFER_FAILED);
    } catch (...) {
        ALGO_LOG_ERROR("detector_self_test unknown exception");
        return static_cast<int>(AlgoError::INFER_FAILED);
    }
}

} // extern "C"
