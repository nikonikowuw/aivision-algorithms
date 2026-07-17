# Runtime module (Apple platform only)
add_library(smoking_runtime STATIC
    apple_buffer_view.mm
    coreml_backend.mm
)

target_include_directories(smoking_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../models
    ${AIVISION_REPO_ROOT}/engine/include
)

target_link_libraries(smoking_runtime PUBLIC
    smoking_common
    smoking_models
)

# Apple 框架
find_library(COREML_FRAMEWORK CoreML)
find_library(ACCELERATE_FRAMEWORK Accelerate)
target_link_libraries(smoking_runtime PRIVATE
    ${COREML_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)
target_compile_options(smoking_runtime PRIVATE -fobjc-arc)
