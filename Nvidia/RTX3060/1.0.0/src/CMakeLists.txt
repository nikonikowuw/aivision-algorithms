# src/CMakeLists.txt
# Unified include path variable — used by all sub-modules
set(ALGO_COMMON_INCLUDES
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
)

add_subdirectory(common)
add_subdirectory(runtime)
add_subdirectory(preprocess)
add_subdirectory(postprocess)
add_subdirectory(models)
add_subdirectory(pipeline)
add_subdirectory(app)
