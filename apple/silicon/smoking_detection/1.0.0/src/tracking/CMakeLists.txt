# Tracking module
add_library(smoking_tracking STATIC
    tracker.cpp
)

target_include_directories(smoking_tracking PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(smoking_tracking PUBLIC
    smoking_common
)
