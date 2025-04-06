cmake_minimum_required(VERSION 3.18)
project(test_deformable_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_plugin test_plugin.cpp)

target_link_libraries(test_plugin
    nvinfer
    nvinfer_plugin
    cudart
)
