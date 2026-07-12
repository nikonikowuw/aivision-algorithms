add_library(algo_runtime STATIC
    inference_backend.cpp
    coreml_backend.mm
)

target_include_directories(algo_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../engine/include
    ${ONNXRUNTIME_INCLUDE_DIR}
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
