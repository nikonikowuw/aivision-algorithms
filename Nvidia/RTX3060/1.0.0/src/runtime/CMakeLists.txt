# runtime/CMakeLists.txt
add_library(algo_runtime STATIC
    trt_backend.cpp
    cuda_utils.cpp
)

target_include_directories(algo_runtime PUBLIC
    ${ALGO_COMMON_INCLUDES}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(algo_runtime PUBLIC
    algo_common
    ${TENSORRT_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NVPARSERS_LIBRARY}
    ${CUDA_LIBRARIES}
)
