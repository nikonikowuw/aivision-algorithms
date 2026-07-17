cmake_minimum_required(VERSION 3.16)
project(safety_helmet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV but we will link modules explicitly to prevent broken viz/hdf dependencies
find_package(OpenCV REQUIRED)

# Homebrew paths for macOS (Apple Silicon)
if(APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
else()
    include_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/onnxruntime/lib)
endif()

# Enable all warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Add subdirectories
add_subdirectory(src)

# Add local test tool executable (linked explicitly only to core and imgcodecs)
add_executable(test_algo tools/test_algo.cpp)
target_include_directories(test_algo PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/../../../../
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(test_algo PRIVATE
    opencv_core opencv_imgcodecs
    dl
)
