add_library(algo_postprocess STATIC
    coordinate.cpp
    scrfd_decoder.cpp
    face_aligner.cpp
)

target_include_directories(algo_postprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
)
