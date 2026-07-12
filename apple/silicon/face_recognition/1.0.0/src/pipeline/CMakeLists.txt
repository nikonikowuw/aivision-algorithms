add_library(algo_pipeline STATIC
    tracker.cpp
    face_index.cpp
    algorithm_context.cpp
)

target_include_directories(algo_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
    ${ONNXRUNTIME_INCLUDE_DIR}
)
