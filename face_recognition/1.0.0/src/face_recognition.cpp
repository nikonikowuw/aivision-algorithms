// 人脸识别算法包 - 主入口
// 实现 ABI 接口，支持 InsightFace + YOLOv8 + ByteTracker

#include "face_recognition.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include <json/json.h>

// ============================================================
// 错误码定义
// ============================================================
enum class AlgoError : int {
    SUCCESS = 0,
    INVALID_HANDLE = -1,
    INVALID_INPUT = -2,
    INVALID_OUTPUT = -3,
    INVALID_SIZE = -4,
    SIZE_TOO_LARGE = -5,
    MEMORY_MAP_FAILED = -6,
    NO_INPUT_DATA = -7,
    INIT_FAILED = -8,
    INFER_FAILED = -9,
    SELF_TEST_FAILED = -10,
};

// ============================================================
// 日志宏
// ============================================================
#define ALGO_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_WARN(fmt, ...)  fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_INFO(fmt, ...)  fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_DEBUG(fmt, ...) // 生产环境禁用 DEBUG

// ============================================================
// 算法上下文 (C++ 内部使用)
// ============================================================
struct AlgorithmContext {
    // 模型实例
    std::unique_ptr<face_recognition::FaceRecognizer> face_recognizer;
    std::unique_ptr<face_recognition::PersonDetector> person_detector;
    std::unique_ptr<face_recognition::ObjectTracker> tracker;
    
    // 配置
    face_recognition::AlgorithmConfig config;
    std::string package_dir;
    
    // 已知人脸库
    std::vector<std::vector<float>> known_embeddings;
    std::vector<std::string> known_ids;
    
    // JSON writer (复用)
    Json::StreamWriterBuilder writer_builder;
    
    AlgorithmContext() {
        writer_builder["indentation"] = "";  // 紧凑格式
    }
};

// ============================================================
// 辅助函数声明 (C++ 内部使用)
// ============================================================
static cv::Rect ExpandBBox(const cv::Rect& bbox, const cv::Size& image_size);
static cv::Mat ConvertInputToMat(const void* input);

// ============================================================
// ABI 接口定义 - 所有导出函数必须在 extern "C" 块中
// ============================================================
extern "C" {

// 项目当前使用的接口 (abi_contract.h)
typedef void* algo_handle_t;

struct hw_buffer_desc_t {
    int dma_fd;            // DMA 文件描述符 (-1 表示不使用)
    size_t size;           // 缓冲区大小
    uint32_t width;        // 图像宽度
    uint32_t height;       // 图像高度
    uint32_t pixel_format; // 像素格式
    int dma_buf_fd;        // DMA-BUF 文件描述符 (可选)
    uint64_t phys_addr;    // 物理地址 (可选)
    void* user_data;       // 用户数据指针 (用于普通内存)
};

struct infer_result_t {
    char* result_json;      // 结果 JSON (由算法 malloc，由调用者 free)
    size_t result_json_len; // JSON 长度
    uint32_t infer_time_us; // 推理耗时 (微秒)
    int reserved[4];        // 预留字段
};

// PRD 定义的接口 (向后兼容)
struct NikoImageFrame {
    const unsigned char* data;
    int width;
    int height;
    int channels;
    const char* pixel_format;
    int stride;
    int dma_fd;
    void* user_data;
};

// ============================================================
// 接口实现 - 项目当前使用 (detector_init/infer/destroy)
// ============================================================

/// 初始化算法上下文
/// @param config_json 配置 JSON 字符串
/// @return 算法句柄，失败返回 nullptr
algo_handle_t detector_init(const char* config_json) {
    if (!config_json) {
        ALGO_LOG_ERROR("detector_init: config_json is null");
        return nullptr;
    }
    
    try {
        auto ctx = std::make_unique<AlgorithmContext>();
        
        // 解析配置
        Json::Value config;
        Json::CharReaderBuilder builder;
        std::string errors;
        std::istringstream config_stream(config_json);
        
        if (!Json::parseFromStream(builder, config_stream, &config, &errors)) {
            ALGO_LOG_ERROR("detector_init: Failed to parse config: %s", errors.c_str());
            return nullptr;
        }
        
        // 设置配置（带验证）
        ctx->config.conf_thres = std::max(0.0f, std::min(1.0f, 
            config.get("conf_thres", 0.5f).asFloat()));
        ctx->config.iou_thres = std::max(0.0f, std::min(1.0f, 
            config.get("iou_thres", 0.45f).asFloat()));
        ctx->config.enable_tracker = config.get("enable_tracker", true).asBool();
        ctx->config.face_conf_thres = std::max(0.0f, std::min(1.0f, 
            config.get("face_conf_thres", 0.6f).asFloat()));
        ctx->config.face_quality_thres = std::max(0.0f, std::min(1.0f, 
            config.get("face_quality_thres", 0.3f).asFloat()));
        ctx->config.max_faces = std::max(1, std::min(100, 
            config.get("max_faces", 10).asInt()));
        ctx->config.recognition_threshold = std::max(0.0f, std::min(1.0f, 
            config.get("recognition_threshold", 0.6f).asFloat()));
        
        // 获取包目录
        ctx->package_dir = config.get("package_dir", "").asString();
        if (ctx->package_dir.empty()) {
            ALGO_LOG_ERROR("detector_init: package_dir not specified");
            return nullptr;
        }
        
        // 初始化人脸检测器
        ctx->face_recognizer = std::make_unique<face_recognition::InsightFaceRecognizer>();
        if (!ctx->face_recognizer->Initialize(ctx->package_dir + "/models/insightface")) {
            ALGO_LOG_ERROR("detector_init: Failed to initialize face recognizer");
            return nullptr;
        }
        
        // 初始化人体检测器
        ctx->person_detector = std::make_unique<face_recognition::YOLOv8Detector>();
        if (!ctx->person_detector->Initialize(ctx->package_dir + "/models/yolov8n.onnx")) {
            ALGO_LOG_ERROR("detector_init: Failed to initialize person detector");
            return nullptr;
        }
        
        // 初始化追踪器（如果启用）
        if (ctx->config.enable_tracker) {
            ctx->tracker = std::make_unique<face_recognition::ByteTracker>();
            if (!ctx->tracker->Initialize()) {
                ALGO_LOG_ERROR("detector_init: Failed to initialize tracker");
                return nullptr;
            }
        }
        
        // 加载已知人脸库（如果有）
        std::string embeddings_path = ctx->package_dir + "/models/embeddings.json";
        std::ifstream embeddings_file(embeddings_path);
        if (embeddings_file.is_open()) {
            try {
                Json::Value embeddings_json;
                embeddings_file >> embeddings_json;
                
                for (const auto& person : embeddings_json) {
                    std::string id = person.get("id", "").asString();
                    std::vector<float> embedding;
                    
                    if (person.isMember("embedding") && person["embedding"].isArray()) {
                        for (const auto& val : person["embedding"]) {
                            embedding.push_back(val.asFloat());
                        }
                    }
                    
                    if (!embedding.empty() && embedding.size() == 512) {
                        ctx->known_embeddings.push_back(embedding);
                        ctx->known_ids.push_back(id);
                    }
                }
                
                ALGO_LOG_INFO("detector_init: Loaded %zu known faces", 
                             ctx->known_embeddings.size());
            } catch (const std::exception& e) {
                ALGO_LOG_WARN("detector_init: Failed to parse embeddings.json: %s", e.what());
            }
        }
        
        ALGO_LOG_INFO("detector_init: Initialization successful");
        return ctx.release();
        
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_init: Exception: %s", e.what());
        return nullptr;
    }
}

/// 执行推理
/// @param handle 算法句柄
/// @param input 输入帧描述符
/// @param context_json 上下文 JSON (可选)
/// @param result 输出结果
/// @return 0 成功，非 0 失败
int detector_infer(algo_handle_t handle,
                   const hw_buffer_desc_t* input,
                   const char* context_json,
                   infer_result_t* result) {
    // 参数验证
    if (!handle) {
        ALGO_LOG_ERROR("detector_infer: Invalid handle");
        return static_cast<int>(AlgoError::INVALID_HANDLE);
    }
    if (!input) {
        ALGO_LOG_ERROR("detector_infer: Invalid input");
        return static_cast<int>(AlgoError::INVALID_INPUT);
    }
    if (!result) {
        ALGO_LOG_ERROR("detector_infer: Invalid output");
        return static_cast<int>(AlgoError::INVALID_OUTPUT);
    }
    if (input->width == 0 || input->height == 0) {
        ALGO_LOG_ERROR("detector_infer: Invalid size: %ux%u", input->width, input->height);
        return static_cast<int>(AlgoError::INVALID_SIZE);
    }
    if (input->width > 8192 || input->height > 8192) {
        ALGO_LOG_ERROR("detector_infer: Size too large: %ux%u", input->width, input->height);
        return static_cast<int>(AlgoError::SIZE_TOO_LARGE);
    }
    
    // 初始化输出
    result->result_json = nullptr;
    result->result_json_len = 0;
    result->infer_time_us = 0;
    
    auto* ctx = static_cast<AlgorithmContext*>(handle);
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 转换输入为 OpenCV Mat
        cv::Mat image = ConvertInputToMat(input);
        if (image.empty()) {
            ALGO_LOG_ERROR("detector_infer: Failed to convert input to image");
            return static_cast<int>(AlgoError::NO_INPUT_DATA);
        }
        
        // 检测人体
        auto persons = ctx->person_detector->DetectPersons(
            image, ctx->config.conf_thres, ctx->config.iou_thres);
        
        // 应用追踪器（如果启用）
        if (ctx->config.enable_tracker && ctx->tracker) {
            persons = ctx->tracker->Update(persons, image);
        }
        
        // 对每个人体区域进行人脸识别
        Json::Value results(Json::arrayValue);
        int face_count = 0;
        
        for (const auto& person : persons) {
            if (face_count >= ctx->config.max_faces) break;
            
            cv::Rect expanded_bbox = ExpandBBox(person.bbox, image.size());
            auto faces = ctx->face_recognizer->DetectFaces(image(expanded_bbox));
            
            for (const auto& face : faces) {
                if (face_count >= ctx->config.max_faces) break;
                if (face.quality_score < ctx->config.face_quality_thres) continue;
                
                cv::Rect face_bbox = face.bbox;
                face_bbox.x += expanded_bbox.x;
                face_bbox.y += expanded_bbox.y;
                
                auto embedding = ctx->face_recognizer->ExtractEmbedding(image, face_bbox, face.landmarks);
                
                Json::Value face_result;
                face_result["detect_confidence"] = face.confidence;
                face_result["quality_score"] = face.quality_score;
                face_result["track_id"] = person.track_id;
                
                Json::Value bbox;
                bbox["x"] = face_bbox.x; bbox["y"] = face_bbox.y;
                bbox["w"] = face_bbox.width; bbox["h"] = face_bbox.height;
                face_result["bbox"] = bbox;
                
                Json::Value landmarks(Json::arrayValue);
                for (const auto& point : face.landmarks) {
                    Json::Value p; p["x"] = point.x; p["y"] = point.y;
                    landmarks.append(p);
                }
                face_result["landmarks"] = landmarks;
                
                if (!embedding.empty()) {
                    float best_similarity = 0.0f;
                    int best_index = -1;
                    
                    for (size_t i = 0; i < ctx->known_embeddings.size(); ++i) {
                        float sim = ctx->face_recognizer->CalculateSimilarity(embedding, ctx->known_embeddings[i]);
                        if (sim > best_similarity) {
                            best_similarity = sim;
                            best_index = static_cast<int>(i);
                        }
                    }
                    
                    if (best_similarity >= ctx->config.recognition_threshold && best_index >= 0) {
                        face_result["category_code"] = 13001;
                        face_result["identity_id"] = ctx->known_ids[best_index];
                    } else {
                        face_result["category_code"] = 13002;
                        face_result["identity_id"] = "";
                    }
                    face_result["similarity"] = best_similarity;
                } else {
                    face_result["category_code"] = 13003;
                    face_result["identity_id"] = "";
                    face_result["similarity"] = 0.0f;
                }
                
                results.append(face_result);
                face_count++;
            }
        }
        
        // 计算推理时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        // 构建返回结果
        Json::Value output;
        output["results"] = results;
        output["infer_time_us"] = static_cast<uint32_t>(duration.count());
        output["frame_width"] = input->width;
        output["frame_height"] = input->height;
        output["face_count"] = face_count;
        
        // 序列化为 JSON 字符串
        std::string json_string = Json::writeString(ctx->writer_builder, output);
        
        // 分配内存并复制结果
        result->result_json = strdup(json_string.c_str());
        if (!result->result_json) {
            ALGO_LOG_ERROR("detector_infer: Failed to allocate memory for result");
            return static_cast<int>(AlgoError::MEMORY_MAP_FAILED);
        }
        result->result_json_len = json_string.size();
        result->infer_time_us = static_cast<uint32_t>(duration.count());
        
        ALGO_LOG_DEBUG("detector_infer: Success, %d faces detected in %u us", 
                      face_count, result->infer_time_us);
        return static_cast<int>(AlgoError::SUCCESS);
        
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_infer: Exception: %s", e.what());
        return static_cast<int>(AlgoError::INFER_FAILED);
    }
}

/// 释放推理结果内存
/// @param result 推理结果
void detector_free_result(infer_result_t* result) {
    if (result && result->result_json) {
        free(result->result_json);  // 匹配 strdup 的 malloc
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

/// 销毁算法上下文
/// @param handle 算法句柄
void detector_destroy(algo_handle_t handle) {
    if (handle) {
        auto* ctx = static_cast<AlgorithmContext*>(handle);
        delete ctx;
        ALGO_LOG_INFO("detector_destroy: Context destroyed");
    }
}

// ============================================================
// 可选符号
// ============================================================

/// 获取算法版本
const char* detector_version() {
    return "1.0.0";
}

/// 获取算法名称
const char* detector_name() {
    return "face_recognition";
}

/// 执行自检
/// @return 0 自检通过，非 0 自检失败
int detector_self_test() {
    ALGO_LOG_INFO("detector_self_test: Starting self test...");
    
    try {
        // 从环境变量获取算法包目录
        const char* pkg_dir_env = getenv("ALGO_PACKAGE_DIR");
        std::string package_dir = pkg_dir_env ? pkg_dir_env : ".";
        
        // 1. 读取并解析 label_map.json
        std::string label_map_path = package_dir + "/label_map.json";
        std::ifstream label_file(label_map_path);
        if (!label_file.is_open()) {
            ALGO_LOG_ERROR("detector_self_test: Cannot open label_map.json at %s", 
                          label_map_path.c_str());
            return static_cast<int>(AlgoError::SELF_TEST_FAILED);
        }
        
        Json::Value label_map;
        label_file >> label_map;
        
        // 收集所有有效的 category_code
        std::set<int> valid_category_codes;
        for (const auto& item : label_map) {
            if (item.isMember("category_code") && item["category_code"].isInt()) {
                int code = item["category_code"].asInt();
                if (code >= 10000 && code <= 99999) {
                    valid_category_codes.insert(code);
                }
            }
        }
        
        if (valid_category_codes.empty()) {
            ALGO_LOG_ERROR("detector_self_test: No valid category_code in label_map.json");
            return static_cast<int>(AlgoError::SELF_TEST_FAILED);
        }
        
        ALGO_LOG_INFO("detector_self_test: Found %zu valid category codes", 
                     valid_category_codes.size());
        
        // 2. 读取 testimage.jpg
        std::string test_image_path = package_dir + "/testimage.jpg";
        cv::Mat test_image = cv::imread(test_image_path, cv::IMREAD_COLOR);
        if (test_image.empty()) {
            ALGO_LOG_ERROR("detector_self_test: Cannot read testimage.jpg at %s", 
                          test_image_path.c_str());
            return static_cast<int>(AlgoError::SELF_TEST_FAILED);
        }
        
        ALGO_LOG_INFO("detector_self_test: Test image loaded: %dx%d", 
                     test_image.cols, test_image.rows);
        
        // 3. 初始化人脸检测器（如果模型存在）
        auto face_recognizer = std::make_unique<face_recognition::InsightFaceRecognizer>();
        std::string face_model_path = package_dir + "/models/insightface";
        
        if (face_recognizer->Initialize(face_model_path)) {
            ALGO_LOG_INFO("detector_self_test: Face recognizer initialized");
            
            // 4. 执行一次推理
            auto faces = face_recognizer->DetectFaces(test_image);
            ALGO_LOG_INFO("detector_self_test: Detected %zu faces", faces.size());
            
            // 5. 验证输出格式
            for (size_t i = 0; i < faces.size(); ++i) {
                const auto& face = faces[i];
                
                // 检查边界框有效性
                if (face.bbox.width <= 0 || face.bbox.height <= 0) {
                    ALGO_LOG_ERROR("detector_self_test: Invalid bbox at face %zu", i);
                    return static_cast<int>(AlgoError::SELF_TEST_FAILED);
                }
                
                // 检查置信度范围
                if (face.confidence < 0.0f || face.confidence > 1.0f) {
                    ALGO_LOG_ERROR("detector_self_test: Invalid confidence at face %zu: %.2f", 
                                  i, face.confidence);
                    return static_cast<int>(AlgoError::SELF_TEST_FAILED);
                }
                
                // 检查关键点数量
                if (face.landmarks.size() != 5) {
                    ALGO_LOG_ERROR("detector_self_test: Invalid landmarks count at face %zu: %zu", 
                                  i, face.landmarks.size());
                    return static_cast<int>(AlgoError::SELF_TEST_FAILED);
                }
            }
            
            // 6. 测试特征提取（如果有人脸）
            if (!faces.empty()) {
                auto embedding = face_recognizer->ExtractEmbedding(
                    test_image, faces[0].bbox, faces[0].landmarks);
                
                // 检查特征向量维度
                if (embedding.size() != 512) {
                    ALGO_LOG_ERROR("detector_self_test: Invalid embedding dimension: %zu", 
                                  embedding.size());
                    return static_cast<int>(AlgoError::SELF_TEST_FAILED);
                }
                
                // 检查特征向量是否归一化
                float norm = 0.0f;
                for (float val : embedding) {
                    norm += val * val;
                }
                norm = std::sqrt(norm);
                if (std::abs(norm - 1.0f) > 0.01f) {
                    ALGO_LOG_WARN("detector_self_test: Embedding not normalized, norm=%.4f", norm);
                }
                
                ALGO_LOG_INFO("detector_self_test: Embedding extraction successful");
            }
        } else {
            ALGO_LOG_WARN("detector_self_test: Face recognizer not initialized (models may not exist)");
        }
        
        // 7. 测试人体检测器（如果模型存在）
        auto person_detector = std::make_unique<face_recognition::YOLOv8Detector>();
        std::string yolov8_path = package_dir + "/models/yolov8n.onnx";
        
        if (person_detector->Initialize(yolov8_path)) {
            ALGO_LOG_INFO("detector_self_test: Person detector initialized");
            
            auto persons = person_detector->DetectPersons(test_image, 0.5f, 0.45f);
            ALGO_LOG_INFO("detector_self_test: Detected %zu persons", persons.size());
            
            for (size_t i = 0; i < persons.size(); ++i) {
                if (persons[i].bbox.width <= 0 || persons[i].bbox.height <= 0) {
                    ALGO_LOG_ERROR("detector_self_test: Invalid person bbox at %zu", i);
                    return static_cast<int>(AlgoError::SELF_TEST_FAILED);
                }
            }
        } else {
            ALGO_LOG_WARN("detector_self_test: Person detector not initialized (models may not exist)");
        }
        
        // 8. 测试追踪器
        auto tracker = std::make_unique<face_recognition::ByteTracker>();
        if (tracker->Initialize()) {
            // 模拟追踪测试
            std::vector<face_recognition::PersonDetection> test_detections;
            face_recognition::PersonDetection det;
            det.bbox = cv::Rect(100, 100, 50, 80);
            det.confidence = 0.9f;
            det.track_id = -1;
            test_detections.push_back(det);
            
            auto tracked = tracker->Update(test_detections, test_image);
            if (tracked.empty() || tracked[0].track_id < 0) {
                ALGO_LOG_ERROR("detector_self_test: Tracker not working");
                return static_cast<int>(AlgoError::SELF_TEST_FAILED);
            }
            
            ALGO_LOG_INFO("detector_self_test: Tracker test passed, track_id=%d", 
                         tracked[0].track_id);
        }
        
        ALGO_LOG_INFO("detector_self_test: All tests passed!");
        return static_cast<int>(AlgoError::SUCCESS);
        
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("detector_self_test: Exception: %s", e.what());
        return static_cast<int>(AlgoError::SELF_TEST_FAILED);
    }
}

// ============================================================
// PRD 定义的接口 (向后兼容)
// ============================================================

/// 创建检测器 (PRD 接口)
/// @param package_dir 算法包目录
/// @return 检测器句柄
void* create_detector(const char* package_dir) {
    if (!package_dir) {
        ALGO_LOG_ERROR("create_detector: package_dir is null");
        return nullptr;
    }
    
    // 构建配置 JSON
    Json::Value config;
    config["package_dir"] = package_dir;
    config["conf_thres"] = 0.5;
    config["iou_thres"] = 0.45;
    config["enable_tracker"] = true;
    config["face_conf_thres"] = 0.6;
    config["face_quality_thres"] = 0.3;
    config["max_faces"] = 10;
    config["recognition_threshold"] = 0.6;
    
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    std::string config_json = Json::writeString(writer, config);
    
    return detector_init(config_json.c_str());
}

/// 执行推理 (PRD 接口)
/// @param detector 检测器句柄
/// @param image_array 图像帧
/// @param ai_params_json 推理参数 JSON
/// @return 结果 JSON 字符串 (由算法 malloc，由调用者调用 detector_free_result 释放)
char* detector_infer_v2(void* detector, const NikoImageFrame* image_array, 
                        const char* ai_params_json) {
    if (!detector || !image_array) {
        ALGO_LOG_ERROR("detector_infer_v2: Invalid parameters");
        return nullptr;
    }
    
    // 转换为 hw_buffer_desc_t
    hw_buffer_desc_t input;
    input.dma_fd = image_array->dma_fd;
    input.size = image_array->width * image_array->height * image_array->channels;
    input.width = image_array->width;
    input.height = image_array->height;
    input.pixel_format = 0;
    input.dma_buf_fd = -1;
    input.phys_addr = 0;
    input.user_data = const_cast<unsigned char*>(image_array->data);
    
    infer_result_t result;
    int ret = detector_infer(detector, &input, ai_params_json, &result);
    
    if (ret == 0 && result.result_json) {
        return result.result_json;
    }
    
    return nullptr;
}

/// 释放结果 (PRD 接口)
/// @param result 结果字符串
void detector_free_result_v2(char* result) {
    if (result) {
        free(result);
    }
}

/// 销毁检测器 (PRD 接口)
/// @param detector 检测器句柄
void destroy_detector(void* detector) {
    detector_destroy(detector);
}

} // extern "C"

// ============================================================
// 辅助函数实现 (C++ 内部使用)
// ============================================================

/// 扩展边界框
static cv::Rect ExpandBBox(const cv::Rect& bbox, const cv::Size& image_size) {
    float expand_ratio = 0.2f;
    int expand_w = static_cast<int>(bbox.width * expand_ratio);
    int expand_h = static_cast<int>(bbox.height * expand_ratio);
    
    cv::Rect expanded(
        std::max(0, bbox.x - expand_w),
        std::max(0, bbox.y - expand_h),
        bbox.width + 2 * expand_w,
        bbox.height + 2 * expand_h
    );
    
    return expanded & cv::Rect(0, 0, image_size.width, image_size.height);
}

/// 将输入转换为 OpenCV Mat
/// @param input 输入描述符 (hw_buffer_desc_t*)
static cv::Mat ConvertInputToMat(const void* input) {
    if (!input) return cv::Mat();
    
    auto* desc = static_cast<const hw_buffer_desc_t*>(input);
    cv::Mat image;
    
    if (desc->dma_fd >= 0) {
        // 硬件缓冲区：通过 mmap 映射
        ALGO_LOG_DEBUG("ConvertInputToMat: Using DMA buffer, fd=%d", desc->dma_fd);
        
        void* mapped = mmap(nullptr, desc->size, PROT_READ, MAP_SHARED, desc->dma_fd, 0);
        if (mapped == MAP_FAILED) {
            ALGO_LOG_ERROR("ConvertInputToMat: mmap failed for fd=%d", desc->dma_fd);
            return cv::Mat();
        }
        
        // 创建 Mat 并拷贝数据（因为 munmap 后数据不可用）
        image = cv::Mat(desc->height, desc->width, CV_8UC3, mapped).clone();
        munmap(mapped, desc->size);
        
    } else if (desc->user_data) {
        // 普通内存：直接使用指针
        ALGO_LOG_DEBUG("ConvertInputToMat: Using user_data pointer");
        image = cv::Mat(desc->height, desc->width, CV_8UC3, 
                       static_cast<uint8_t*>(desc->user_data)).clone();
    }
    
    return image;
}
