cmake_minimum_required(VERSION 3.16)
project(gpu_face_recognition VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- CUDA ----
find_package(CUDA REQUIRED)
enable_language(CUDA)

# ---- TensorRT (manual find) ----
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h PATH_SUFFIXES NvInfer)
find_library(TENSORRT_LIBRARY NAMES nvinfer)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin)
find_library(NVPARSERS_LIBRARY NAMES nvparsers)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found! Please check TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")

# ---- CUDA NPP (for face alignment) ----
find_library(CUDA_NPP_LIBRARY NAMES nppig nppc nppial)
if(NOT CUDA_NPP_LIBRARY)
    message(WARNING "CUDA NPP library not found — GPU face alignment may be limited.")
endif()

# ---- FAISS (optional, for gallery matching) ----
find_path(FAISS_INCLUDE_DIR NAMES faiss/Index.h)
find_library(FAISS_LIBRARY NAMES faiss)
find_library(BLASFEO_LIBRARY NAMES openblas blas)
find_library(LAPACK_LIBRARY NAMES lapack)

if(FAISS_LIBRARY)
    set(HAS_FAISS TRUE)
    message(STATUS "FAISS found: ${FAISS_LIBRARY}")
else()
    set(HAS_FAISS FALSE)
    message(WARNING "FAISS not found — gallery matching will use brute-force only.")
endif()

# ---- Include paths ----
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

if(FAISS_INCLUDE_DIR)
    include_directories(${FAISS_INCLUDE_DIR})
endif()

# ---- Compiler flags ----
add_compile_options(-Wall -Wextra -Wpedantic)

# ---- CUDA architectures (RTX 3060 = SM 86) ----
set(CMAKE_CUDA_ARCHITECTURES 86)

# ---- Build sub-modules ----
add_subdirectory(src)

# ---- Tests ----
include(CTest)
if(BUILD_TESTING)
    add_executable(gpu_face_recognition_unit_tests tests/test_core.cpp)
    target_include_directories(gpu_face_recognition_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(gpu_face_recognition_unit_tests PRIVATE
        algo_preprocess
        algo_pipeline
        algo_common
    )
    add_test(NAME gpu_face_recognition_unit_tests COMMAND gpu_face_recognition_unit_tests)
endif()
