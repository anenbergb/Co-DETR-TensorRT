cmake_minimum_required(VERSION 3.18)
project(test_deformable_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorRT
if(NOT DEFINED TENSORRT_DIR)
    message(FATAL_ERROR "TENSORRT_DIR not specified. Use -DTENSORRT_DIR=/path/to/TensorRT")
endif()

add_executable(test_plugin test_plugin.cpp)

target_link_libraries(test_plugin
    ${TENSORRT_DIR}/lib/libnvinfer.so
    ${TENSORRT_DIR}/lib/libnvinfer_plugin.so
    cudart
)
