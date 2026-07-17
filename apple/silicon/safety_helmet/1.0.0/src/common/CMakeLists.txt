add_library(algo_common STATIC
    config_parser.cpp
    input_buffer_validator.cpp
)

target_include_directories(algo_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
