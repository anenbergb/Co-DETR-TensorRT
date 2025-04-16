cmake_minimum_required(VERSION 3.18)
project(test_deformable_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries:    ${CUDAToolkit_LIBRARY_DIR}")

# Allow user to pass TENSORRT_DIR, or use default
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT lib directory")
message(STATUS "TensorRT libs:    ${TENSORRT_LIB_DIR}")

find_library(TENSORRT_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} REQUIRED)

add_executable(test_plugin test_plugin.cpp)

target_link_libraries(test_plugin
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart
)
