if(APPLE)
    add_library(tentcoo_detection SHARED algo_api.mm)
    set_source_files_properties(algo_api.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    add_library(tentcoo_detection SHARED algo_api.cpp)
endif()

target_link_libraries(tentcoo_detection PRIVATE
    algo_pipeline
    algo_models
    algo_postprocess
    algo_preprocess
    algo_runtime
    algo_common
    opencv_core opencv_imgproc opencv_imgcodecs
)

target_include_directories(tentcoo_detection PRIVATE
    ${ALGO_COMMON_INCLUDES}
)

set_target_properties(tentcoo_detection PROPERTIES
    PREFIX ""
    OUTPUT_NAME "tentcoo_detection"
    SUFFIX ".so"
)

# Copy the generated library to the project root after build
add_custom_command(TARGET tentcoo_detection POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tentcoo_detection> ${PROJECT_SOURCE_DIR}/tentcoo_detection.so
)
