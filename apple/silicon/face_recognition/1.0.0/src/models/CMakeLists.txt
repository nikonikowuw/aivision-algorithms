add_library(algo_models STATIC
    scrfd_model.cpp
    adaface_model.cpp
)

target_include_directories(algo_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)
