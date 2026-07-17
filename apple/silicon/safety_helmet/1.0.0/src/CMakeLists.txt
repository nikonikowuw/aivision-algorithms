# src/CMakeLists.txt
set(ALGO_COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../
)

add_subdirectory(common)
add_subdirectory(preprocess)
add_subdirectory(postprocess)
add_subdirectory(runtime)
add_subdirectory(models)
add_subdirectory(pipeline)
add_subdirectory(app)
