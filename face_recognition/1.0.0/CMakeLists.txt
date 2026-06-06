cmake_minimum_required(VERSION 3.16)
project(face_recognition VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(ONNXRuntime REQUIRED)

# 添加源文件
add_library(nikoniko_detector SHARED
    src/face_recognition.cpp
    src/yolov8_detector.cpp
    src/insightface_recognizer.cpp
    src/byte_tracker.cpp
)

# 包含目录
target_include_directories(nikoniko_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(nikoniko_detector PRIVATE
    ${OpenCV_LIBS}
    ONNXRuntime::ONNXRuntime
)

# 设置输出名称
set_target_properties(nikoniko_detector PROPERTIES
    OUTPUT_NAME "nikoniko_detector"
    PREFIX ""
    SUFFIX ".so"
)

# 安装
install(TARGETS nikoniko_detector
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)