# common/CMakeLists.txt
add_library(algo_common STATIC
    logger.cpp
    config.cpp
)

target_include_directories(algo_common PUBLIC
    ${ALGO_COMMON_INCLUDES}
)
