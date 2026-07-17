# DAMO-YOLO model wrapper (hardware-independent backend contract).
add_library(smoking_models STATIC
    damoyolo_detector.cpp
)

target_include_directories(smoking_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(smoking_models PUBLIC
    smoking_common
    smoking_postprocess
)

