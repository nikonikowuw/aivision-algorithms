# postprocess/CMakeLists.txt
add_library(algo_postprocess STATIC
    yolo_decoder.cpp
    retina_decoder.cpp
    face_aligner.cpp
    coordinate.cpp
)

target_include_directories(algo_postprocess PUBLIC
    ${ALGO_COMMON_INCLUDES}
)

target_link_libraries(algo_postprocess PUBLIC
    algo_common
)
