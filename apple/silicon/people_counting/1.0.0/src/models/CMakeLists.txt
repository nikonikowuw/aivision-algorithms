add_library(algo_models STATIC
    yolov8_detector.mm
)

target_include_directories(algo_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
)
