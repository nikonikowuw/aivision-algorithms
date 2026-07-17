add_library(algo_pipeline STATIC
    safety_helmet_pipeline.cpp
)

target_include_directories(algo_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(algo_pipeline PUBLIC
    opencv_core opencv_imgproc
)
