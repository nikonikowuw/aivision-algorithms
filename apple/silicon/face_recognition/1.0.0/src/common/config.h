/**
 * @file config.h
 * @brief 人脸识别算法包配置管理接口
 *        Configuration management interface for the face recognition package.
 * @module 基础配置层 (Config Layer)
 * @details 采用三级配置优先级：编译默认值 < .env 文件 < 运行时 config_json。
 *          后一级覆盖前一级，确保灵活部署的同时提供合理的默认值。
 *          模型权重路径、算法阈值、跟踪参数、日志级别等均可通过配置控制。
 */

#ifndef FACE_RECOGNITION_CONFIG_H
#define FACE_RECOGNITION_CONFIG_H

#include <string>
#include <unordered_map>

namespace face_rec {

/**
 * @struct Config
 * @brief 算法全局配置结构 (Global algorithm configuration)
 * @details 包含模型路径、检测/识别阈值、跟踪参数、日志级别等所有可配置项。
 *          通过 LoadConfig() 静态方法创建，内部自动处理三级加载过程。
 */
struct Config {
    std::string package_dir;
    
    std::string scrfd_person_model_path = "weights/scrfd_person_2.5g.mlpackage";
    std::string scrfd_500m_model_path = "weights/scrfd_2.5g_bnkps_shape160x160.mlpackage";
    std::string adaface_model_path = "weights/adaface_ir101_webface4m.mlpackage";
    
    // Config parameters
    bool enable_tracker = true;
    float person_conf_thres = 0.45f;
    float face_conf_thres = 0.55f;
    float recognition_threshold = 0.45f;
    bool zero_copy_required = true;
    bool allow_cpu_fallback = false;

    /**
     * @struct HeadROIParams
     * @brief 头部 ROI（感兴趣区域）推断参数
     *        Head Region-of-Interest inference parameters.
     * @details 根据人体检测框的宽高比（h/w）推断头部所占比例，
     *          用于从人体框中精确裁剪出面部区域进行检测。
     *          站姿（full body）、半身（half body）、蹲姿（crouch）
     *          三种姿态使用不同的头部比例参数。
     */
    struct HeadROIParams {
        int min_body_height = 80;         // Minimum body height (pixels) to attempt head inference
        float full_body_ratio = 0.30f;    // Head ratio when h/w >= 2.5 (full body standing)
        float half_body_ratio = 0.45f;    // Head ratio when 1.5 <= h/w < 2.5 (half body)
        float crouch_ratio = 0.55f;       // Head ratio when h/w < 1.5 (crouching / seated)
        float width_expand = 0.15f;       // Lateral expansion ratio around the estimated head region
    } head_roi;

    float nms_iou_thres = 0.45f;
    float tracker_iou_thres = 0.30f;
    int tracker_max_lost_frames = 30;
    int max_persons = 50;
    int max_faces = 50;
    int thread_pool_size = 4;
    int ort_intra_op_threads = 2;

    // Log levels map (Module Name -> Level Name)
    std::unordered_map<std::string, std::string> log_levels;

    // Timing & Debug configs
    bool enable_timing_report = true;
    int log_timing_interval = 100;
    bool log_model_io = false;

    // Functions

    /**
     * @brief 加载配置（三级优先级）
     *        Load configuration with 3-tier priority.
     * @details 加载顺序：
     *          1. 编译默认值（结构体初始化）
     *          2. .env 文件（与动态库同目录，键值对覆盖）
     *          3. 运行时 config_json 字符串（JSON 格式，最高优先级）
     * @param config_json 运行时 JSON 配置字符串，可为 nullptr
     * @return 合并后的完整配置对象
     */
    static Config LoadConfig(const char* config_json);
};

/**
 * @name 路径与文件工具函数
 * @brief 跨平台路径操作辅助函数
 *        Portable path manipulation utilities.
 */
//@{
std::string DirName(const std::string& path);        // Get directory name from a path (like dirname(1))
std::string JoinPath(const std::string& base, const std::string& path);  // Join base directory with relative path
bool FileExists(const std::string& path);              // Check if a file exists on disk
//@}

} // namespace face_rec

#endif // FACE_RECOGNITION_CONFIG_H
