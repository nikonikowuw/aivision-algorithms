add_library(algo_models STATIC
    damoyolo_model.cpp
)

target_include_directories(algo_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
