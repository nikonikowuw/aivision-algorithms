add_library(algo_common STATIC
    config_parser.cpp
)

target_include_directories(algo_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
