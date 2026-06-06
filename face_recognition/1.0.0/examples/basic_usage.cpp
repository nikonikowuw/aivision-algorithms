// 基本使用示例

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <dlfcn.h>

// 定义ABI接口
typedef void* algo_handle_t;

struct hw_buffer_desc_t {
    int dma_fd;
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int dma_buf_fd;
    uint64_t phys_addr;
};

struct infer_result_t {
    char* result_json;
    size_t result_json_len;
    uint32_t infer_time_us;
    int reserved[4];
};

// 函数指针类型
typedef algo_handle_t (*detector_init_func)(const char* config_json);
typedef int (*detector_infer_func)(algo_handle_t handle,
                                  const hw_buffer_desc_t* input,
                                  const char* context_json,
                                  infer_result_t* result);
typedef void (*detector_free_result_func)(infer_result_t* result);
typedef void (*detector_destroy_func)(algo_handle_t handle);
typedef const char* (*detector_version_func)(void);
typedef const char* (*detector_name_func)(void);
typedef int (*detector_self_test_func)(void);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <算法包路径>" << std::endl;
        return 1;
    }
    
    std::string package_path = argv[1];
    std::string so_path = package_path + "/nikoniko_detector.so";
    
    // 加载动态库
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "无法加载动态库: " << dlerror() << std::endl;
        return 1;
    }
    
    // 获取函数指针
    auto init_func = (detector_init_func)dlsym(handle, "detector_init");
    auto infer_func = (detector_infer_func)dlsym(handle, "detector_infer");
    auto free_result_func = (detector_free_result_func)dlsym(handle, "detector_free_result");
    auto destroy_func = (detector_destroy_func)dlsym(handle, "detector_destroy");
    auto version_func = (detector_version_func)dlsym(handle, "detector_version");
    auto name_func = (detector_name_func)dlsym(handle, "detector_name");
    auto self_test_func = (detector_self_test_func)dlsym(handle, "detector_self_test");
    
    if (!init_func || !infer_func || !free_result_func || !destroy_func) {
        std::cerr << "缺少必需的符号" << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // 打印算法信息
    if (version_func) {
        std::cout << "算法版本: " << version_func() << std::endl;
    }
    if (name_func) {
        std::cout << "算法名称: " << name_func() << std::endl;
    }
    
    // 执行自检
    if (self_test_func) {
        std::cout << "执行自检..." << std::endl;
        int test_result = self_test_func();
        if (test_result == 0) {
            std::cout << "自检通过" << std::endl;
        } else {
            std::cerr << "自检失败: " << test_result << std::endl;
        }
    }
    
    // 初始化算法
    std::string config_json = R"({
        "package_dir": ")" + package_path + R"(",
        "conf_thres": 0.5,
        "iou_thres": 0.45,
        "enable_tracker": true,
        "face_conf_thres": 0.6,
        "face_quality_thres": 0.3,
        "max_faces": 10
    })";
    
    algo_handle_t algo_handle = init_func(config_json.c_str());
    if (!algo_handle) {
        std::cerr << "初始化算法失败" << std::endl;
        dlclose(handle);
        return 1;
    }
    
    std::cout << "算法初始化成功" << std::endl;
    
    // 创建测试图像数据
    // 注意：这里使用的是模拟数据，实际使用时需要真实的图像数据
    const int width = 640;
    const int height = 480;
    const int channels = 3;
    
    // 分配图像内存
    std::vector<uint8_t> image_data(width * height * channels, 128);
    
    // 设置硬件缓冲区描述符
    hw_buffer_desc_t input;
    input.dma_fd = -1;
    input.size = image_data.size();
    input.width = width;
    input.height = height;
    input.pixel_format = 0; // RGB
    input.dma_buf_fd = -1;
    input.phys_addr = 0;
    
    // 执行推理
    infer_result_t result;
    int ret = infer_func(algo_handle, &input, nullptr, &result);
    
    if (ret == 0) {
        std::cout << "推理成功" << std::endl;
        std::cout << "推理时间: " << result.infer_time_us << " 微秒" << std::endl;
        std::cout << "结果JSON: " << result.result_json << std::endl;
        
        // 释放结果
        free_result_func(&result);
    } else {
        std::cerr << "推理失败: " << ret << std::endl;
    }
    
    // 销毁算法
    destroy_func(algo_handle);
    
    // 关闭动态库
    dlclose(handle);
    
    std::cout << "完成" << std::endl;
    
    return 0;
}