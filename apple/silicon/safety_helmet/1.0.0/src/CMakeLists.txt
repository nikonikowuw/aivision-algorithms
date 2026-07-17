# src/CMakeLists.txt
set(ALGO_COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/src
)

add_subdirectory(common)
add_subdirectory(preprocess)
add_subdirectory(postprocess)
add_subdirectory(runtime)
add_subdirectory(models)
add_subdirectory(pipeline)
add_subdirectory(app)
