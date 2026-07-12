add_library(algo_models STATIC
    scrfd_model.cpp
    adaface_model.cpp
)

target_include_directories(algo_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
