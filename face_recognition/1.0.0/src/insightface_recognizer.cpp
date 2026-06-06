#include "face_recognition.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace face_recognition {

class InsightFaceRecognizer : public FaceRecognizer {
public:
    InsightFaceRecognizer() = default;
    ~InsightFaceRecognizer() override = default;
    
    bool Initialize(const std::string& model_dir) override {
        try {
            // 初始化ONNX Runtime
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "InsightFace");
            session_options_ = std::make_unique<Ort::SessionOptions>();
            session_options_->SetIntraOpNumThreads(1);
            session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 加载人脸检测模型
            std::string det_model_path = model_dir + "/det_10g.onnx";
            det_session_ = std::make_unique<Ort::Session>(*env_, det_model_path.c_str(), *session_options_);
            
            // 加载人脸特征提取模型
            std::string rec_model_path = model_dir + "/w600k_r50.onnx";
            rec_session_ = std::make_unique<Ort::Session>(*env_, rec_model_path.c_str(), *session_options_);
            
            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            
            // 检测模型输入输出
            det_input_name_ = det_session_->GetInputName(0, allocator);
            det_output_name_ = det_session_->GetOutputName(0, allocator);
            
            // 识别模型输入输出
            rec_input_name_ = rec_session_->GetInputName(0, allocator);
            rec_output_name_ = rec_session_->GetOutputName(0, allocator);
            
            // 获取输入形状
            auto input_tensor_info = det_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto input_shape = input_tensor_info.GetShape();
            det_input_height_ = static_cast<int>(input_shape[2]);
            det_input_width_ = static_cast<int>(input_shape[3]);
            
            auto rec_input_tensor_info = rec_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto rec_input_shape = rec_input_tensor_info.GetShape();
            rec_input_height_ = static_cast<int>(rec_input_shape[2]);
            rec_input_width_ = static_cast<int>(rec_input_shape[3]);
            
            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "InsightFace initialization failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<FaceDetection> DetectFaces(const cv::Mat& image) override {
        if (!initialized_ || image.empty()) {
            return {};
        }
        
        // 预处理图像
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(det_input_width_, det_input_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        
        // 创建输入tensor
        std::vector<int64_t> input_shape = {1, 3, det_input_height_, det_input_width_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            reinterpret_cast<float*>(resized.data),
            resized.total() * sizeof(float),
            input_shape.data(), 
            input_shape.size()
        );
        
        // 运行推理
        auto output_tensors = det_session_->Run(
            Ort::RunOptions{nullptr},
            &det_input_name_,
            &input_tensor,
            1,
            &det_output_name_,
            1
        );
        
        // 解析结果
        auto& output_tensor = output_tensors[0];
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        float* output_data = output_tensor.GetTensorMutableData<float>();
        
        // 解析检测结果
        std::vector<FaceDetection> faces;
        int num_detections = static_cast<int>(output_shape[1]);
        
        for (int i = 0; i < num_detections; ++i) {
            float* detection = output_data + i * 16; // 每个检测16个值
            
            float confidence = detection[14];
            if (confidence < 0.5f) continue;
            
            FaceDetection face;
            face.confidence = confidence;
            
            // 解析边界框
            float x1 = detection[0] * image.cols;
            float y1 = detection[1] * image.rows;
            float x2 = detection[2] * image.cols;
            float y2 = detection[3] * image.rows;
            
            face.bbox = cv::Rect(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(x2 - x1),
                static_cast<int>(y2 - y1)
            );
            
            // 解析关键点
            for (int j = 0; j < 5; ++j) {
                float kx = detection[4 + j * 2] * image.cols;
                float ky = detection[5 + j * 2] * image.rows;
                face.landmarks.emplace_back(kx, ky);
            }
            
            // 计算质量评分
            face.quality_score = CalculateFaceQuality(image, face.bbox, face.landmarks);
            
            faces.push_back(face);
        }
        
        return faces;
    }
    
    std::vector<float> ExtractEmbedding(const cv::Mat& image,
                                       const cv::Rect& face_bbox,
                                       const std::vector<cv::Point2f>& landmarks) override {
        if (!initialized_ || image.empty()) return {};
        
        cv::Mat aligned_face = AlignFace(image, face_bbox, landmarks);
        cv::Mat resized;
        cv::resize(aligned_face, resized, cv::Size(rec_input_width_, rec_input_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        
        std::vector<int64_t> input_shape = {1, 3, rec_input_height_, rec_input_width_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(resized.data),
            resized.total() * sizeof(float), input_shape.data(), input_shape.size());
        
        auto output_tensors = rec_session_->Run(Ort::RunOptions{nullptr}, &rec_input_name_, &input_tensor, 1, &rec_output_name_, 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> embedding(output_data, output_data + output_size);
        float norm = std::sqrt(std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
        if (norm > 0) {
            for (float& val : embedding) val /= norm;
        }
        return embedding;
    }
    
    float CalculateSimilarity(const std::vector<float>& embedding1,
                             const std::vector<float>& embedding2) override {
        if (embedding1.size() != embedding2.size() || embedding1.empty()) return 0.0f;
        return std::inner_product(embedding1.begin(), embedding1.end(), embedding2.begin(), 0.0f);
    }
    
    std::vector<FaceRecognitionResult> RecognizeFaces(
        const cv::Mat& image,
        const std::vector<std::vector<float>>& known_embeddings,
        const std::vector<std::string>& known_ids,
        float similarity_threshold) override {
        
        std::vector<FaceRecognitionResult> results;
        
        // 检测人脸
        auto faces = DetectFaces(image);
        
        for (const auto& face : faces) {
            FaceRecognitionResult result;
            result.bbox = face.bbox;
            result.detect_confidence = face.confidence;
            result.landmarks = face.landmarks;
            result.quality_score = face.quality_score;
            result.track_id = -1;
            
            // 提取特征
            auto embedding = ExtractEmbedding(image, face.bbox, face.landmarks);
            
            if (embedding.empty()) {
                result.category_code = 13003; // 人员
                result.identity_id = "";
                result.similarity = 0.0f;
            } else {
                // 在已知人脸库中搜索
                float best_similarity = 0.0f;
                int best_index = -1;
                
                for (size_t i = 0; i < known_embeddings.size(); ++i) {
                    float similarity = CalculateSimilarity(embedding, known_embeddings[i]);
                    if (similarity > best_similarity) {
                        best_similarity = similarity;
                        best_index = static_cast<int>(i);
                    }
                }
                
                if (best_similarity >= similarity_threshold && best_index >= 0) {
                    result.category_code = 13001; // 已识别人员
                    result.identity_id = known_ids[best_index];
                    result.similarity = best_similarity;
                } else {
                    result.category_code = 13002; // 未知人员
                    result.identity_id = "";
                    result.similarity = best_similarity;
                }
            }
            
            results.push_back(result);
        }
        
        return results;
    }
    
private:
    float CalculateFaceQuality(const cv::Mat& image, const cv::Rect& bbox,
                              const std::vector<cv::Point2f>& landmarks) {
        if (bbox.width <= 0 || bbox.height <= 0) {
            return 0.0f;
        }
        
        // 检查边界框是否在图像内
        cv::Rect image_rect(0, 0, image.cols, image.rows);
        cv::Rect valid_bbox = bbox & image_rect;
        
        if (valid_bbox.width <= 0 || valid_bbox.height <= 0) {
            return 0.0f;
        }
        
        // 计算人脸区域占比
        float face_ratio = static_cast<float>(valid_bbox.area()) / static_cast<float>(image.total());
        
        // 计算清晰度（使用Laplacian方差）
        cv::Mat face_roi = image(valid_bbox);
        cv::Mat gray;
        cv::cvtColor(face_roi, gray, cv::COLOR_BGR2GRAY);
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        cv::Scalar mean, stddev;
        cv::meanStdDev(laplacian, mean, stddev);
        float clarity = static_cast<float>(stddev[0] * stddev[0]);
        
        // 归一化到0-1
        float quality = std::min(1.0f, clarity / 1000.0f);
        
        return quality;
    }
    
    cv::Mat AlignFace(const cv::Mat& image, const cv::Rect& bbox,
                     const std::vector<cv::Point2f>& landmarks) {
        // 使用关键点进行人脸对齐
        if (landmarks.size() < 5) {
            // 如果没有关键点，直接裁剪
            cv::Rect safe_bbox = bbox & cv::Rect(0, 0, image.cols, image.rows);
            return image(safe_bbox).clone();
        }
        
        // 计算仿射变换
        std::vector<cv::Point2f> src_points = {
            landmarks[0], // 左眼
            landmarks[1], // 右眼
            landmarks[2]  // 鼻子
        };
        
        // 目标位置（标准人脸位置，基于112x112输出）
        std::vector<cv::Point2f> dst_points = {
            cv::Point2f(38.2946f, 51.6963f), // 左眼
            cv::Point2f(73.5318f, 51.5014f), // 右眼
            cv::Point2f(56.0252f, 71.7366f)  // 鼻子
        };
        
        // 计算变换矩阵
        cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
        
        // 应用变换
        cv::Mat aligned;
        cv::warpAffine(image, aligned, transform, cv::Size(112, 112));
        
        return aligned;
    }
    
    bool initialized_ = false;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> det_session_;
    std::unique_ptr<Ort::Session> rec_session_;
    
    std::string det_input_name_;
    std::string det_output_name_;
    std::string rec_input_name_;
    std::string rec_output_name_;
    
    int det_input_height_ = 640;
    int det_input_width_ = 640;
    int rec_input_height_ = 112;
    int rec_input_width_ = 112;
};

} // namespace face_recognition