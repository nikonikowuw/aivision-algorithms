cmake_minimum_required(VERSION 3.16)
project(face_recognition VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    execute_process(
        COMMAND xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE MACOSX_SDKROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(MACOSX_SDKROOT AND EXISTS "${MACOSX_SDKROOT}/usr/include/c++/v1/string")
        add_compile_options(-isystem ${MACOSX_SDKROOT}/usr/include/c++/v1)
    endif()
endif()

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(ONNXRuntime REQUIRED)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(JSONCPP jsoncpp)
endif()

# 添加源文件
add_library(nikoniko_detector SHARED
    src/face_recognition.cpp
    src/yolov11_detector.cpp
    src/insightface_recognizer.cpp
    src/byte_tracker.cpp
)

# 包含目录
target_include_directories(nikoniko_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 链接库
target_link_libraries(nikoniko_detector PRIVATE
    ${OpenCV_LIBS}
)

if(TARGET ONNXRuntime::ONNXRuntime)
    target_link_libraries(nikoniko_detector PRIVATE ONNXRuntime::ONNXRuntime)
elseif(TARGET onnxruntime::onnxruntime)
    target_link_libraries(nikoniko_detector PRIVATE onnxruntime::onnxruntime)
elseif(ONNXRuntime_LIBRARIES)
    target_link_libraries(nikoniko_detector PRIVATE ${ONNXRuntime_LIBRARIES})
elseif(ONNXRUNTIME_LIBRARIES)
    target_link_libraries(nikoniko_detector PRIVATE ${ONNXRUNTIME_LIBRARIES})
else()
    find_library(ONNXRUNTIME_LIBRARY onnxruntime)
    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNXRuntime library not found")
    endif()
    target_link_libraries(nikoniko_detector PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()

if(JSONCPP_FOUND)
    target_link_directories(nikoniko_detector PRIVATE ${JSONCPP_LIBRARY_DIRS})
    target_link_options(nikoniko_detector PRIVATE ${JSONCPP_LDFLAGS_OTHER})
    target_link_libraries(nikoniko_detector PRIVATE ${JSONCPP_LIBRARIES})
else()
    find_library(JSONCPP_LIBRARY jsoncpp)
    if(NOT JSONCPP_LIBRARY)
        message(FATAL_ERROR "jsoncpp library not found")
    endif()
    target_link_libraries(nikoniko_detector PRIVATE ${JSONCPP_LIBRARY})
endif()

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