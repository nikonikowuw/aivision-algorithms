cmake_minimum_required(VERSION 3.16)
project(fall_detection VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED)

# ONNXRuntime 的 Homebrew/官方预编译包通常不提供统一的 CMake package config。
# 因此这里优先使用 find_package，找不到时回退到头文件与动态库手动查找。
find_package(ONNXRuntime QUIET)

if(NOT TARGET ONNXRuntime::ONNXRuntime)
    set(_ORT_HINTS "")
    if(ONNXRuntime_ROOT)
        list(APPEND _ORT_HINTS "${ONNXRuntime_ROOT}")
    endif()
    if(onnxruntime_ROOT)
        list(APPEND _ORT_HINTS "${onnxruntime_ROOT}")
    endif()
    if(DEFINED ENV{ONNXRuntime_ROOT})
        list(APPEND _ORT_HINTS "$ENV{ONNXRuntime_ROOT}")
    endif()
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        list(APPEND _ORT_HINTS "$ENV{ONNXRUNTIME_ROOT}")
    endif()
    list(APPEND _ORT_HINTS
        "/opt/homebrew/opt/onnxruntime"
        "/usr/local/opt/onnxruntime"
        "/opt/onnxruntime"
    )

    find_path(ONNXRuntime_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${_ORT_HINTS}
        PATH_SUFFIXES
            include
            include/onnxruntime
            include/onnxruntime/core/session
    )

    find_library(ONNXRuntime_LIBRARY
        NAMES onnxruntime libonnxruntime
        HINTS ${_ORT_HINTS}
        PATH_SUFFIXES lib lib64
    )

    if(NOT ONNXRuntime_INCLUDE_DIR OR NOT ONNXRuntime_LIBRARY)
        message(FATAL_ERROR
            "ONNXRuntime C++ SDK not found.\n"
            "Install it first, for example on macOS:\n"
            "  brew install onnxruntime\n"
            "Or provide a SDK path containing include/ and lib/:\n"
            "  cmake -S . -B build -DONNXRuntime_ROOT=/path/to/onnxruntime\n"
            "Current include result: ${ONNXRuntime_INCLUDE_DIR}\n"
            "Current library result: ${ONNXRuntime_LIBRARY}"
        )
    endif()

    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
    set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}"
    )
    set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_INCLUDE_DIR}")
endif()

add_library(nikoniko_detector SHARED
    src/fall_detection.cpp
)

# 部分 macOS CommandLineTools 版本不会自动搜索 SDK 内的 libc++ 头文件目录，
# 会导致第三方头文件 include <cmath> 失败；显式加入该目录以保证可构建。
if(APPLE AND CMAKE_OSX_SYSROOT AND EXISTS "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1/cmath")
    target_include_directories(nikoniko_detector SYSTEM BEFORE PRIVATE
        "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1"
    )
endif()

target_include_directories(nikoniko_detector PRIVATE
    ../../../engine/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
)

target_link_libraries(nikoniko_detector PRIVATE
    ${OpenCV_LIBS}
    ONNXRuntime::ONNXRuntime
)

target_compile_options(nikoniko_detector PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

set_target_properties(nikoniko_detector PROPERTIES
    PREFIX ""
    OUTPUT_NAME "nikoniko_detector"
    SUFFIX ".so"
)

install(TARGETS nikoniko_detector
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
)
