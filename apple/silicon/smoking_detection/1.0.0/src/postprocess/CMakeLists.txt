# Postprocess module
add_library(smoking_postprocess STATIC
    damoyolo_decoder.cpp
)

target_include_directories(smoking_postprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(smoking_postprocess PUBLIC
    smoking_common
)
