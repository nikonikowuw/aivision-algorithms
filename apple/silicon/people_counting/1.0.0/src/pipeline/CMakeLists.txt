add_library(algo_pipeline STATIC
    tracker.cpp
    counting.cpp
    pipeline.cpp
)

target_include_directories(algo_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
)
