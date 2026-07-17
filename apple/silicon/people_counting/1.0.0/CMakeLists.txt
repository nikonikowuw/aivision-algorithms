cmake_minimum_required(VERSION 3.16)
project(people_counting VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Homebrew paths for macOS (Apple Silicon)
if(APPLE)
    include_directories(/opt/homebrew/include)
    link_directories(/opt/homebrew/lib)
endif()

# Enable all warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Add subdirectories
add_subdirectory(src)
