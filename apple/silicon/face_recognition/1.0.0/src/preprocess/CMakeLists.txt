add_library(algo_preprocess STATIC
    image_utils.cpp
)

target_include_directories(algo_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
)
