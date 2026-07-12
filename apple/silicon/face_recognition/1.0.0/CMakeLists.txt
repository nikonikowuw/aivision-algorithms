cmake_minimum_required(VERSION 3.16)
project(face_recognition VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



# Homebrew paths for macOS (Apple Silicon)
if(APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

# Find ONNX Runtime (manually locate using find_path/find_library to be robust)
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_c_api.h PATH_SUFFIXES onnxruntime)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found! Please check ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY.")
endif()

# Enable all warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Add subdirectories
add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
    add_executable(face_recognition_unit_tests tests/test_core.cpp)
    target_include_directories(face_recognition_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(face_recognition_unit_tests PRIVATE
        algo_preprocess
        algo_pipeline
        algo_common
    )
    add_test(NAME face_recognition_unit_tests COMMAND face_recognition_unit_tests)
endif()
