find_package(OpenCV REQUIRED)

add_library(algo_preprocess STATIC
    letterbox.cpp
)

target_include_directories(algo_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(algo_preprocess PUBLIC
    opencv_core opencv_imgproc
)
