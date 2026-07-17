# models/CMakeLists.txt
add_library(algo_models STATIC
    yolo11_model.cpp
    retinaface_model.cpp
    adaface_model.cpp
)

target_include_directories(algo_models PUBLIC
    ${ALGO_COMMON_INCLUDES}
)

target_link_libraries(algo_models PUBLIC
    algo_common
    algo_runtime
    algo_preprocess
    algo_postprocess
)
