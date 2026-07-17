# Common module
add_library(smoking_common STATIC
    config.cpp
    event_json.cpp
    logger.cpp
)

target_include_directories(smoking_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party
)
