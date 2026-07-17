add_library(people_counting SHARED
    algo_api.mm
)

target_link_libraries(people_counting PRIVATE
    algo_pipeline
    algo_models
    algo_postprocess
    algo_preprocess
    algo_runtime
    algo_common
)

target_include_directories(people_counting PRIVATE
    ${ALGO_COMMON_INCLUDES}
)

set_target_properties(people_counting PROPERTIES
    PREFIX ""
    OUTPUT_NAME "people_counting"
    SUFFIX ".so"
)

# Copy the generated library to the project root after build
add_custom_command(TARGET people_counting POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:people_counting> ${PROJECT_SOURCE_DIR}/people_counting.so
)
