add_library(algo_runtime STATIC
    coreml_backend.mm
)

target_include_directories(algo_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ALGO_COMMON_INCLUDES}
)

if(APPLE)
    find_library(COREML_FRAMEWORK CoreML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(algo_runtime PUBLIC 
        ${COREML_FRAMEWORK} 
        ${FOUNDATION_FRAMEWORK}
    )
endif()
