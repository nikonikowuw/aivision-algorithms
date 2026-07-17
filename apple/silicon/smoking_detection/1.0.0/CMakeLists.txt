cmake_minimum_required(VERSION 3.16)
project(smoking_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(AIVISION_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(SMOKING_RUNTIME_DIR "${CMAKE_BINARY_DIR}/runtime")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

enable_testing()

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")

    find_program(XCRUN_EXECUTABLE xcrun REQUIRED)
    set(METAL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/image_preprocess.metal")
    set(METAL_AIR "${CMAKE_BINARY_DIR}/shaders/image_preprocess.air")
    set(METALLIB_OUTPUT "${SMOKING_RUNTIME_DIR}/shaders/image_preprocess.metallib")
    add_custom_command(
        OUTPUT "${METALLIB_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SMOKING_RUNTIME_DIR}/shaders"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metal -c "${METAL_SOURCE}" -o "${METAL_AIR}"
        COMMAND "${XCRUN_EXECUTABLE}" -sdk macosx metallib "${METAL_AIR}" -o "${METALLIB_OUTPUT}"
        DEPENDS "${METAL_SOURCE}"
        VERBATIM)
    add_custom_target(smoking_metallib ALL DEPENDS "${METALLIB_OUTPUT}")
endif()

add_subdirectory(src)
add_subdirectory(tests)

if(APPLE)
    add_dependencies(tentcoo_detection smoking_metallib)
    add_custom_command(TARGET tentcoo_detection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SMOKING_RUNTIME_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/algo_meta.yaml" "${SMOKING_RUNTIME_DIR}/algo_meta.yaml"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/label_map.json" "${SMOKING_RUNTIME_DIR}/label_map.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/README.md" "${SMOKING_RUNTIME_DIR}/README.md"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/testimage.jpg" "${SMOKING_RUNTIME_DIR}/testimage.jpg"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/weights" "${SMOKING_RUNTIME_DIR}/weights"
        VERBATIM)

    add_executable(smoking_test_tool tools/test_algo.mm)
    target_include_directories(smoking_test_tool PRIVATE
        "${AIVISION_REPO_ROOT}/engine/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    target_compile_options(smoking_test_tool PRIVATE -fobjc-arc)
    set_target_properties(smoking_test_tool PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME smoking_runtime_self_test
             COMMAND smoking_test_tool "${SMOKING_RUNTIME_DIR}/tentcoo_detection.so")
endif()
