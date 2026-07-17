cmake_minimum_required(VERSION 3.16)
project(safety_helmet VERSION 1.0.0 LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV but we will link modules explicitly to prevent broken viz/hdf dependencies
find_package(OpenCV REQUIRED)

# ONNX Runtime detection.
# On macOS, prefer Homebrew; on Linux, use bundled third_party.
# Set ONNXRUNTIME_ROOT to override.
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(ONNXRUNTIME_ROOT)
    include_directories(${ONNXRUNTIME_ROOT}/include)
    link_directories(${ONNXRUNTIME_ROOT}/lib)
elseif(APPLE)
    # macOS: detect ONNX Runtime via Homebrew or common paths.
    # Try brew --prefix first, then fall back to known locations.
    execute_process(
        COMMAND brew --prefix onnxruntime
        OUTPUT_VARIABLE ONNX_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(ONNX_BREW_PREFIX AND EXISTS "${ONNX_BREW_PREFIX}/include/onnxruntime_cxx_api.h")
        include_directories(${ONNX_BREW_PREFIX}/include)
        link_directories(${ONNX_BREW_PREFIX}/lib)
    elseif(EXISTS /opt/homebrew/include/onnxruntime_cxx_api.h)
        include_directories(/opt/homebrew/include)
        link_directories(/opt/homebrew/lib)
    elseif(EXISTS /opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h)
        include_directories(/opt/homebrew/include/onnxruntime)
        link_directories(/opt/homebrew/lib)
    elseif(EXISTS /usr/local/include/onnxruntime_cxx_api.h)
        include_directories(/usr/local/include)
        link_directories(/usr/local/lib)
    else()
        message(FATAL_ERROR
            "onnxruntime not found. Install it via: brew install onnxruntime\n"
            "Or set ONNXRUNTIME_ROOT=/path/to/onnxruntime")
    endif()
else()
    # Linux / other: use bundled third_party.
    include_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib)
endif()

# Enable all warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Add subdirectories
add_subdirectory(src)

enable_testing()

add_executable(test_input_buffer_validator tools/test_input_buffer_validator.cpp)
target_include_directories(test_input_buffer_validator PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(test_input_buffer_validator PRIVATE algo_common)
add_test(NAME input_buffer_validator COMMAND test_input_buffer_validator)

# Add local test tool executable (linked explicitly only to core and imgcodecs)
add_executable(test_algo tools/test_algo.cpp)
target_include_directories(test_algo PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_algo PRIVATE
    opencv_core opencv_imgcodecs
    dl
)
