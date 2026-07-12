add_library(algo_common STATIC
    config.cpp
    logger.cpp
)

target_include_directories(algo_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
)
