#include "pipeline/safety_helmet_pipeline.h"
#include "common/logger.h"
#include "common/timer.h"
#include "engine/include/algo/abi_contract.h"
#include <exception>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <dlfcn.h>

namespace safety_helmet {
namespace pixel_format {
    constexpr uint32_t BGR24 = ('B') | ('G' << 8) | ('R' << 16) | ('3' << 24);
    constexpr uint32_t NV12  = ('N') | ('V' << 8) | ('1' << 16) | ('2' << 24);
}
}

static std::string SerializeDetections(const std::vector<safety_helmet::Detection>& detections) {
    std::stringstream ss;
    ss << "[\n";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        ss << "  {\n";
        ss << "    \"category_code\": " << det.category_code << ",\n";
        ss << "    \"category_name\": \"" << det.category_name << "\",\n";
        ss << "    \"detect_confidence\": " << std::fixed << std::setprecision(4) << det.confidence << ",\n";
        ss << "    \"bbox\": [" 
           << std::fixed << std::setprecision(6) 
           << det.x << ", " 
           << det.y << ", " 
           << det.w << ", " 
           << det.h << "]\n";
        ss << "  }" << (i + 1 < detections.size() ? "," : "") << "\n";
    }
    ss << "]";
    return ss.str();
}

static std::string GetLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&GetLibraryDir), &info) != 0 && info.dli_fname) {
        std::string path(info.dli_fname);
        auto last_slash = path.find_last_of('/');
        if (last_slash == std::string::npos) return ".";
        return path.substr(0, last_slash);
    }
    return ".";
}

extern "C" {

algo_handle_t detector_init(const char *config_json) {
    try {
        if (!config_json) return nullptr;
        auto *pipeline = new safety_helmet::SafetyHelmetPipeline();
        if (!pipeline->Initialize(config_json)) {
            delete pipeline;
            return nullptr;
        }
        return reinterpret_cast<algo_handle_t>(pipeline);
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_init C++ exception: %s", e.what());
        return nullptr;
    } catch (...) {
        ALGO_LOG_ERROR("detector_init unknown C++ exception");
        return nullptr;
    }
}

int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result) {
    try {
        if (!handle || !input || !result) return -1;
        result->result_json = nullptr;
        result->result_json_len = 0;
        result->infer_time_us = 0;

        if (!input->data || input->width <= 0 || input->height <= 0) {
            ALGO_LOG_ERROR("detector_infer: Invalid input parameters");
            return -2;
        }

        safety_helmet::Timer timer;
        timer.start();

        auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);

        cv::Mat frame;
        int frame_w = static_cast<int>(input->width);
        int frame_h = static_cast<int>(input->height);

        if (input->pixel_format == safety_helmet::pixel_format::BGR24) {
            frame = cv::Mat(frame_h, frame_w, CV_8UC3, const_cast<void*>(input->data), input->stride);
        } else if (input->pixel_format == safety_helmet::pixel_format::NV12) {
            cv::Mat nv12_mat(frame_h * 3 / 2, input->stride, CV_8UC1, const_cast<void*>(input->data));
            cv::Mat bgr_mat;
            cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
            frame = bgr_mat(cv::Rect(0, 0, frame_w, frame_h));
        } else {
            ALGO_LOG_ERROR("detector_infer: Unsupported pixel format: 0x%08x", input->pixel_format);
            return -2;
        }

        std::vector<safety_helmet::Detection> detections;
        if (!pipeline->Detect(frame, detections)) {
            ALGO_LOG_ERROR("detector_infer: Detection failed");
            return -3;
        }

        std::string json_str = SerializeDetections(detections);
        result->result_json_len = json_str.size();
        result->result_json = static_cast<char *>(std::malloc(result->result_json_len + 1));
        std::memcpy(result->result_json, json_str.c_str(), result->result_json_len + 1);

        result->infer_time_us = timer.elapsed_us();
        return 0;
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_infer C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOG_ERROR("detector_infer unknown C++ exception");
        return -5;
    }
}

void detector_destroy(algo_handle_t handle) {
    try {
        if (handle) {
            auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);
            pipeline->Destroy();
            delete pipeline;
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_destroy C++ exception: %s", e.what());
    } catch (...) {
        ALGO_LOG_ERROR("detector_destroy unknown C++ exception");
    }
}

const char *detector_version(void) {
    return "1.0.0";
}

const char *detector_name(void) {
    return "safety_helmet";
}

int detector_self_test(void) {
    try {
        if (sizeof(hw_buffer_desc_t) != 144) {
            ALGO_LOG_ERROR("self_test failed: hw_buffer_desc_t size %zu != 144", sizeof(hw_buffer_desc_t));
            return -1;
        }
        if (sizeof(infer_result_t) != 40) {
            ALGO_LOG_ERROR("self_test failed: infer_result_t size %zu != 40", sizeof(infer_result_t));
            return -2;
        }

        std::string lib_dir = GetLibraryDir();
        std::string image_path = safety_helmet::JoinPath(lib_dir, "testimage.jpg");
        std::string model_path = safety_helmet::JoinPath(lib_dir, "weights/damoyolo_safety_helmet.onnx");

        if (!safety_helmet::FileExists(image_path)) {
            ALGO_LOG_ERROR("self_test failed: testimage.jpg not found at %s", image_path.c_str());
            return -3;
        }

        // Create temporary config JSON
        std::stringstream ss;
        ss << "{\n"
           << "  \"package_dir\": \"" << lib_dir << "\",\n"
           << "  \"model_path\": \"" << model_path << "\",\n"
           << "  \"conf_threshold\": 0.5,\n"
           << "  \"iou_threshold\": 0.45\n"
           << "}";

        std::string config_json = ss.str();
        
        algo_handle_t handle = detector_init(config_json.c_str());
        if (!handle) {
            ALGO_LOG_ERROR("self_test failed: detector_init returned null");
            return -4;
        }

        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            ALGO_LOG_ERROR("self_test failed: cv::imread could not load %s", image_path.c_str());
            detector_destroy(handle);
            return -5;
        }

        hw_buffer_desc_t input{};
        input.width = img.cols;
        input.height = img.rows;
        input.pixel_format = safety_helmet::pixel_format::BGR24;
        input.data = img.data;
        input.stride = img.cols * 3;
        input.size = img.cols * img.rows * 3;

        infer_result_t result{};
        int ret = detector_infer(handle, &input, nullptr, &result);
        if (ret != 0) {
            ALGO_LOG_ERROR("self_test failed: detector_infer returned %d", ret);
            detector_destroy(handle);
            return -6;
        }

        ALGO_LOG_INFO("self_test: Inference succeeded. Results:\n%s", result.result_json);
        algo_free_result(&result);

        detector_destroy(handle);
        ALGO_LOG_INFO("self_test passed successfully");
        return 0;
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("self_test C++ exception: %s", e.what());
        return -8;
    } catch (...) {
        ALGO_LOG_ERROR("self_test unknown C++ exception");
        return -8;
    }
}

void algo_free_result(infer_result_t *result) {
    if (result && result->result_json) {
        std::free(result->result_json);
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

void detector_free_result(infer_result_t *result) {
    algo_free_result(result);
}

} // extern "C"
