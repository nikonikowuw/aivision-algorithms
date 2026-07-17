add_library(algo_runtime STATIC
    ort_session_manager.cpp
)

target_include_directories(algo_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(algo_runtime PUBLIC
    onnxruntime
)

if(APPLE)
    find_library(COREML_FRAMEWORK CoreML)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(algo_runtime PUBLIC 
        ${COREML_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
endif()
