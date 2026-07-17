# Preprocess module (Apple platform only)
add_library(smoking_preprocess STATIC
    metal_preprocessor.mm
)

target_include_directories(smoking_preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(smoking_preprocess PUBLIC
    smoking_common
)

# Apple 框架
find_library(METAL_FRAMEWORK Metal)
find_library(COREVIDEO_FRAMEWORK CoreVideo)
find_library(IOSURFACE_FRAMEWORK IOSurface)
target_link_libraries(smoking_preprocess PRIVATE
    ${METAL_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
    ${IOSURFACE_FRAMEWORK}
)
target_compile_options(smoking_preprocess PRIVATE -fobjc-arc)
