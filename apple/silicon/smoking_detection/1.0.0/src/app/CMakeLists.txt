add_library(tentcoo_detection SHARED algo_api.mm)

target_include_directories(tentcoo_detection PRIVATE
    ${AIVISION_REPO_ROOT}/engine/include
    ${PROJECT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime)

target_link_libraries(tentcoo_detection PRIVATE
    smoking_common
    smoking_pipeline
    smoking_runtime)
target_compile_options(tentcoo_detection PRIVATE -fobjc-arc)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(IOSURFACE_FRAMEWORK IOSurface REQUIRED)
target_link_libraries(tentcoo_detection PRIVATE
    ${FOUNDATION_FRAMEWORK}
    ${COREVIDEO_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    ${IMAGEIO_FRAMEWORK}
    ${IOSURFACE_FRAMEWORK})

set_target_properties(tentcoo_detection PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "tentcoo_detection"
    LIBRARY_OUTPUT_DIRECTORY "${SMOKING_RUNTIME_DIR}"
    MACOSX_RPATH ON
    INSTALL_RPATH "@loader_path")
