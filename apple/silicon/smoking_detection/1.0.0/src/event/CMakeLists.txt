# Event state machine module
add_library(smoking_event STATIC
    event_state.cpp
)

target_include_directories(smoking_event PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(smoking_event PUBLIC
    smoking_common
)
