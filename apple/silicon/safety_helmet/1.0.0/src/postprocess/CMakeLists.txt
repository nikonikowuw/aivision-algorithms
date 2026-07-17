add_library(algo_postprocess STATIC
    nms.cpp
    coord_mapper.cpp
    damoyolo_decoder.cpp
)

target_include_directories(algo_postprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(algo_postprocess PUBLIC
    opencv_core opencv_imgproc
)
