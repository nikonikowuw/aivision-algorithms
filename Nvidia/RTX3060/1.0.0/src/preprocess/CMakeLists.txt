# preprocess/CMakeLists.txt
add_library(algo_preprocess STATIC
    image_utils.cpp
)

target_include_directories(algo_preprocess PUBLIC
    ${ALGO_COMMON_INCLUDES}
)

target_link_libraries(algo_preprocess PUBLIC
    algo_common
)
