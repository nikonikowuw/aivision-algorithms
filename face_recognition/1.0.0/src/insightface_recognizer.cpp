#include "face_recognition.h"
#include "ort_coreml.h"
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
            
            // 加载 YOLOv11n-face 人脸检测模型 (MLProgram，100% CoreML，float32)
            {
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(1);
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
                std::unordered_map<std::string,std::string> ep = {
                    {"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"},
                    {"RequireStaticInputShapes", "0"}, {"EnableOnSubgraphs", "1"},
                };
                opts.AppendExecutionProvider("CoreML", ep);
                std::string det_model_path = model_dir + "/../yolov11n-face-fp32.onnx";
                fprintf(stderr, "[INFO] Creating det_session_\n");
                det_session_ = std::make_unique<Ort::Session>(*env_, det_model_path.c_str(), opts);
                fprintf(stderr, "[INFO] det_session_ created\n");
            }

            // 加载 2d106det 人脸对齐关键点模型 (NeuralNetwork)
            {
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(1);
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
                std::unordered_map<std::string,std::string> ep = {
                    {"ModelFormat", "NeuralNetwork"}, {"MLComputeUnits", "ALL"},
                    {"RequireStaticInputShapes", "0"}, {"EnableOnSubgraphs", "1"},
                };
                opts.AppendExecutionProvider("CoreML", ep);
                std::string align_model_path = model_dir + "/2d106det.onnx";
                fprintf(stderr, "[INFO] Creating align_session_\n");
                align_session_ = std::make_unique<Ort::Session>(*env_, align_model_path.c_str(), opts);
                fprintf(stderr, "[INFO] align_session_ created\n");
            }
            
            // 加载 w600k_r50 人脸特征提取模型 (NeuralNetwork，避免float16)
            {
                Ort::SessionOptions opts;
                opts.SetIntraOpNumThreads(1);
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
                std::unordered_map<std::string,std::string> ep = {
                    {"ModelFormat", "NeuralNetwork"}, {"MLComputeUnits", "ALL"},
                    {"RequireStaticInputShapes", "0"}, {"EnableOnSubgraphs", "1"},
                };
                opts.AppendExecutionProvider("CoreML", ep);
                std::string rec_model_path = model_dir + "/w600k_r50.onnx";
                fprintf(stderr, "[INFO] Creating rec_session_\n");
                rec_session_ = std::make_unique<Ort::Session>(*env_, rec_model_path.c_str(), opts);
                fprintf(stderr, "[INFO] rec_session_ created\n");
            }
            
            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            
            // 检测模型输入输出
            fprintf(stderr, "[INFO] Getting det input/output names\n");
            auto det_input_name = det_session_->GetInputNameAllocated(0, allocator);
            auto det_output_name = det_session_->GetOutputNameAllocated(0, allocator);
            det_input_name_ = det_input_name.get();
            det_output_name_ = det_output_name.get();
            fprintf(stderr, "[INFO] det: %s -> %s\n", det_input_name_.c_str(), det_output_name_.c_str());
            
            // 识别模型输入输出
            fprintf(stderr, "[INFO] Getting rec input/output names\n");
            auto rec_input_name = rec_session_->GetInputNameAllocated(0, allocator);
            auto rec_output_name = rec_session_->GetOutputNameAllocated(0, allocator);
            rec_input_name_ = rec_input_name.get();
            rec_output_name_ = rec_output_name.get();
            fprintf(stderr, "[INFO] rec: %s -> %s\n", rec_input_name_.c_str(), rec_output_name_.c_str());
            
            // 获取输入形状
            fprintf(stderr, "[INFO] Getting det input shape\n");
            auto det_input_type = det_session_->GetInputTypeInfo(0);
            auto det_tensor_info = det_input_type.GetTensorTypeAndShapeInfo();
            fprintf(stderr, "[INFO] det tensor type: %d\n", static_cast<int>(det_tensor_info.GetElementType()));
            det_input_type_ = det_tensor_info.GetElementType();
            auto input_shape = det_tensor_info.GetShape();
            fprintf(stderr, "[INFO] det shape dims: %zu\n", input_shape.size());
            for (size_t i = 0; i < input_shape.size(); i++) {
                fprintf(stderr, "[INFO]   dim[%zu] = %lld\n", i, input_shape[i]);
            }
            det_input_height_ = static_cast<int>(input_shape[2]);
            det_input_width_ = static_cast<int>(input_shape[3]);
            fprintf(stderr, "[INFO] det shape: %dx%d\n", det_input_width_, det_input_height_);
            
            fprintf(stderr, "[INFO] Getting rec input shape\n");
            auto rec_input_type = rec_session_->GetInputTypeInfo(0);
            auto rec_tensor_info = rec_input_type.GetTensorTypeAndShapeInfo();
            fprintf(stderr, "[INFO] rec tensor type: %d\n", static_cast<int>(rec_tensor_info.GetElementType()));
            auto rec_input_shape = rec_tensor_info.GetShape();
            fprintf(stderr, "[INFO] rec shape dims: %zu\n", rec_input_shape.size());
            for (size_t i = 0; i < rec_input_shape.size(); i++) {
                fprintf(stderr, "[INFO]   dim[%zu] = %lld\n", i, rec_input_shape[i]);
            }
            rec_input_height_ = static_cast<int>(rec_input_shape[2]);
            rec_input_width_ = static_cast<int>(rec_input_shape[3]);
            fprintf(stderr, "[INFO] rec shape: %dx%d\n", rec_input_width_, rec_input_height_);
            
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
        
        // 创建输入tensor（根据模型输入类型自动处理float16/float32）
        std::vector<int64_t> input_shape = {1, 3, det_input_height_, det_input_width_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<FaceDetection> faces;
        const char* det_input_names[] = {det_input_name_.c_str()};
        const char* det_output_names[] = {det_output_name_.c_str()};
        
        // float32 输入
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(resized.data),
            resized.total() * sizeof(float), input_shape.data(), input_shape.size());
        auto output_tensors = det_session_->Run(
            Ort::RunOptions{nullptr}, det_input_names, &input_tensor, 1, det_output_names, 1);
        
        auto& output_tensor = output_tensors[0];
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        float* output_data = output_tensor.GetTensorMutableData<float>();
        
        // YOLOv8 格式: [1, 5, 8400] -> [batch, 4+bbox+conf, num_detections]
        int num_detections = static_cast<int>(output_shape[2]);
        
        for (int i = 0; i < num_detections; ++i) {
            float cx = output_data[0 * num_detections + i];
            float cy = output_data[1 * num_detections + i];
            float w = output_data[2 * num_detections + i];
            float h = output_data[3 * num_detections + i];
            float confidence = output_data[4 * num_detections + i];
            
            if (confidence < 0.5f) continue;
            
            FaceDetection face;
            face.confidence = confidence;
            face.bbox = cv::Rect(
                static_cast<int>((cx - w/2) * image.cols),
                static_cast<int>((cy - h/2) * image.rows),
                static_cast<int>(w * image.cols),
                static_cast<int>(h * image.rows));
            
            // YOLOv8-face 没有关键点输出，留空
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
        
        const char* rec_input_names[] = {rec_input_name_.c_str()};
        const char* rec_output_names[] = {rec_output_name_.c_str()};
        auto output_tensors = rec_session_->Run(Ort::RunOptions{nullptr}, rec_input_names, &input_tensor, 1, rec_output_names, 1);
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
    std::unique_ptr<Ort::Session> det_session_;
    std::unique_ptr<Ort::Session> align_session_;
    std::unique_ptr<Ort::Session> rec_session_;
    
    std::string det_input_name_;
    std::string det_output_name_;
    std::string rec_input_name_;
    std::string rec_output_name_;
    
    int det_input_height_ = 640;
    int det_input_width_ = 640;
    ONNXTensorElementDataType det_input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    int rec_input_height_ = 112;
    int rec_input_width_ = 112;
};

std::unique_ptr<FaceRecognizer> CreateInsightFaceRecognizer() {
    return std::make_unique<InsightFaceRecognizer>();
}

} // namespace face_recognition