#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace face_recognition {

// 人脸检测结果
struct FaceDetection {
    cv::Rect bbox;
    float confidence;
    std::vector<cv::Point2f> landmarks; // 5个关键点
    float quality_score;
};

// 人脸识别结果
struct FaceRecognitionResult {
    cv::Rect bbox;
    float detect_confidence;
    std::vector<cv::Point2f> landmarks;
    float quality_score;
    int category_code; // 13001: 已识别人员, 13002: 未知人员, 13003: 人员
    std::string identity_id;
    float similarity;
    int track_id;
};

// 人体检测结果
struct PersonDetection {
    cv::Rect bbox;
    float confidence;
    int track_id;
};

// 算法配置
struct AlgorithmConfig {
    float conf_thres = 0.5f;              // 人体检测置信度阈值
    float iou_thres = 0.45f;              // NMS IOU 阈值
    bool enable_tracker = true;           // 是否启用 ByteTracker
    float face_conf_thres = 0.6f;         // 人脸检测置信度阈值
    float face_quality_thres = 0.3f;      // 人脸质量阈值
    int max_faces = 10;                   // 单帧最大人脸数
    float recognition_threshold = 0.6f;   // 人脸识别相似度阈值
};

// 人脸识别器接口
class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;
    
    // 初始化模型
    virtual bool Initialize(const std::string& model_dir) = 0;
    
    // 检测人脸
    virtual std::vector<FaceDetection> DetectFaces(const cv::Mat& image) = 0;
    
    // 提取人脸特征
    virtual std::vector<float> ExtractEmbedding(const cv::Mat& image, 
                                                const cv::Rect& face_bbox,
                                                const std::vector<cv::Point2f>& landmarks) = 0;
    
    // 计算相似度
    virtual float CalculateSimilarity(const std::vector<float>& embedding1,
                                     const std::vector<float>& embedding2) = 0;
    
    // 人脸识别
    virtual std::vector<FaceRecognitionResult> RecognizeFaces(
        const cv::Mat& image,
        const std::vector<std::vector<float>>& known_embeddings,
        const std::vector<std::string>& known_ids,
        float similarity_threshold = 0.6f) = 0;
};

// 人体检测器接口
class PersonDetector {
public:
    virtual ~PersonDetector() = default;
    
    // 初始化模型
    virtual bool Initialize(const std::string& model_path) = 0;
    
    // 检测人体
    virtual std::vector<PersonDetection> DetectPersons(const cv::Mat& image,
                                                      float conf_thres,
                                                      float iou_thres) = 0;
};

// 目标追踪器接口
class ObjectTracker {
public:
    virtual ~ObjectTracker() = default;
    
    // 初始化追踪器
    virtual bool Initialize() = 0;
    
    // 更新追踪
    virtual std::vector<PersonDetection> Update(
        const std::vector<PersonDetection>& detections,
        const cv::Mat& image) = 0;
    
    // 重置追踪器
    virtual void Reset() = 0;
};

} // namespace face_recognition

#endif // FACE_RECOGNITION_H