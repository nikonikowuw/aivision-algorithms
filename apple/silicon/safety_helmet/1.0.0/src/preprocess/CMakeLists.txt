find_package(OpenCV REQUIRED)

add_library(algo_preprocess STATIC
    letterbox.cpp
)

target_include_directories(algo_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(algo_preprocess PUBLIC
    opencv_core opencv_imgproc
)

if(APPLE)
    target_sources(algo_preprocess PRIVATE metal_preprocessor.mm)
    find_library(METAL_FRAMEWORK Metal)
    find_library(COREVIDEO_FRAMEWORK CoreVideo)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(algo_preprocess PUBLIC
        ${METAL_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    # ObjC++ source requires automatic reference counting.
    set_source_files_properties(metal_preprocessor.mm PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )
else()
    # Non-Apple stub: MetalPreprocessor always returns IsAvailable() == false.
    target_sources(algo_preprocess PRIVATE metal_preprocessor_stub.cpp)
endif()
