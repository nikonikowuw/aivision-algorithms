add_library(algo_pipeline STATIC
    tracker.cpp
    face_index.cpp
    thread_pool.cpp
    attribute.cpp
    algorithm_context.cpp
)

target_include_directories(algo_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
    ${ONNXRUNTIME_INCLUDE_DIR}
)
