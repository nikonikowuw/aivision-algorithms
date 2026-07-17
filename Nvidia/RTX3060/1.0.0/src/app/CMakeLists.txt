# app/CMakeLists.txt
add_library(nikoniko_detector SHARED
    abi_entry.cpp
)

target_link_libraries(nikoniko_detector PRIVATE
    algo_pipeline
    algo_models
    algo_postprocess
    algo_preprocess
    algo_runtime
    algo_common
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
)

target_include_directories(nikoniko_detector PRIVATE
    ${ALGO_COMMON_INCLUDES}
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(nikoniko_detector PROPERTIES
    PREFIX ""
    OUTPUT_NAME "nikoniko_detector"
    SUFFIX ".so"
)

# Copy the generated library to the project root after build
add_custom_command(TARGET nikoniko_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nikoniko_detector> ${PROJECT_SOURCE_DIR}/nikoniko_detector.so
)
