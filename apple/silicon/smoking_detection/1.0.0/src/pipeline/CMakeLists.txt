add_library(smoking_pipeline STATIC pipeline.cpp)

target_include_directories(smoking_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../event
    ${CMAKE_CURRENT_SOURCE_DIR}/../models
    ${CMAKE_CURRENT_SOURCE_DIR}/../postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/../preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/../tracking)

target_link_libraries(smoking_pipeline PUBLIC
    smoking_common
    smoking_event
    smoking_models
    smoking_postprocess
    smoking_preprocess
    smoking_runtime
    smoking_tracking)
