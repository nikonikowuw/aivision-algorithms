#include "face_recognition.h"
#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <algorithm>

namespace face_recognition {

class YOLOv8Detector : public PersonDetector {
public:
    YOLOv8Detector() = default;
    ~YOLOv8Detector() override = default;
    
    bool Initialize(const std::string& model_path) override {
        try {
            // 初始化ONNX Runtime
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
            session_options_ = std::make_unique<Ort::SessionOptions>();
            session_options_->SetIntraOpNumThreads(1);
            session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 加载模型
            session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
            
            // 获取输入输出名称
            Ort::AllocatorWithDefaultOptions allocator;
            input_name_ = session_->GetInputName(0, allocator);
            output_name_ = session_->GetOutputName(0, allocator);
            
            // 获取输入形状
            auto input_tensor_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto input_shape = input_tensor_info.GetShape();
            input_height_ = static_cast<int>(input_shape[2]);
            input_width_ = static_cast<int>(input_shape[3]);
            
            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "YOLOv8 initialization failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::vector<PersonDetection> DetectPersons(const cv::Mat& image,
                                              float conf_thres,
                                              float iou_thres) override {
        if (!initialized_ || image.empty()) return {};
        
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_width_, input_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        
        std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, reinterpret_cast<float*>(resized.data),
            resized.total() * sizeof(float), input_shape.data(), input_shape.size());
        
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, &input_name_, &input_tensor, 1, &output_name_, 1);
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        
        int num_detections = static_cast<int>(output_shape[2]);
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        
        float scale_x = static_cast<float>(image.cols) / input_width_;
        float scale_y = static_cast<float>(image.rows) / input_height_;
        
        for (int i = 0; i < num_detections; ++i) {
            float* det = output_data + i;
            float score = det[4 * num_detections]; // Class 0 (Person)
            
            if (score >= conf_thres) {
                float cx = det[0 * num_detections];
                float cy = det[1 * num_detections];
                float w = det[2 * num_detections];
                float h = det[3 * num_detections];
                
                boxes.emplace_back(
                    static_cast<int>((cx - w/2) * scale_x),
                    static_cast<int>((cy - h/2) * scale_y),
                    static_cast<int>(w * scale_x),
                    static_cast<int>(h * scale_y)
                );
                confidences.push_back(score);
            }
        }
        
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thres, iou_thres, indices);
        
        std::vector<PersonDetection> results;
        for (int idx : indices) {
            results.push_back({boxes[idx], confidences[idx], -1});
        }
        return results;
    }
    
private:
    
    bool initialized_ = false;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    std::string input_name_;
    std::string output_name_;
    
    int input_height_ = 640;
    int input_width_ = 640;
};

} // namespace face_recognition