/**
 * @file config.h
 * @brief GPU Face Recognition — Configuration management interface.
 * @module Config Layer
 * @details 3-tier priority: compile defaults < .env file < runtime config_json.
 */

#ifndef GPU_FACE_RECOGNITION_CONFIG_H
#define GPU_FACE_RECOGNITION_CONFIG_H

#include <string>
#include <unordered_map>

namespace face_rec {

struct Config {
    std::string package_dir;

    // TensorRT engine paths
    std::string yolo11n_engine_path = "weights/yolo11n.engine";
    std::string retinaface_engine_path = "weights/retinaface.engine";
    std::string adaface_ir50_engine_path = "weights/adaface_ir50.engine";

    // Detection thresholds
    bool enable_tracker = true;
    float person_conf_thres = 0.45f;
    float face_conf_thres = 0.55f;
    float recognition_threshold = 0.45f;

    // NMS
    float nms_iou_thres = 0.45f;

    // Tracker
    float tracker_iou_thres = 0.30f;
    int tracker_max_lost_frames = 30;

    // Batch scheduling
    int max_batch_size = 16;
    int batch_timeout_ms = 8;
    int max_persons = 50;
    int max_faces = 50;

    // CUDA / TensorRT
    int cuda_device_id = 0;
    size_t trt_max_workspace_size = 2ULL * 1024 * 1024 * 1024;  // 2GB

    // Log levels map (Module Name -> Level Name)
    std::unordered_map<std::string, std::string> log_levels;

    // Timing & Debug
    bool enable_timing_report = true;
    int log_timing_interval = 100;
    bool log_model_io = false;

    /**
     * @brief Load configuration with 3-tier priority.
     * @param config_json Runtime JSON config string (can be nullptr)
     * @return Merged configuration object
     */
    static Config LoadConfig(const char* config_json);
};

// Path utilities
std::string DirName(const std::string& path);
std::string JoinPath(const std::string& base, const std::string& path);
bool FileExists(const std::string& path);

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_CONFIG_H
