cmake_minimum_required(VERSION 3.18)
project(codetr_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries:    ${CUDAToolkit_LIBRARY_DIR}")

# Dependencies
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)

# argparse (header-only)
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

# Allow user to pass TENSORRT_DIR, or use default
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT lib directory")
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include directory")

message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libs:    ${TENSORRT_LIB_DIR}")

find_library(TENSORRT_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} REQUIRED)

# Allow user to pass TORCHTRT_DIR, or use default
set(TORCHTRT_DIR "/usr/local/lib/python3.12/dist-packages/torch_tensorrt" CACHE PATH "Path to Torch-TensorRT installation")
message(STATUS "Using Torch-TensorRT from: ${TORCHTRT_DIR}")

find_library(TORCHTRT_LIB torchtrt PATHS ${TORCHTRT_DIR}/lib REQUIRED)
find_library(TORCHTRT_RUNTIME_LIB torchtrt_runtime PATHS ${TORCHTRT_DIR}/lib REQUIRED)
find_library(TORCHTRT_PLUGINS_LIB torchtrt_plugins PATHS ${TORCHTRT_DIR}/lib REQUIRED)

# Add the executable
add_executable(codetr_inference codetr_inference.cpp)

# Include paths
target_include_directories(codetr_inference PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${argparse_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCHVISION_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${TORCHTRT_DIR}/include
)

# Add linker flags before Torch-TensorRT libs
target_link_options(codetr_inference PRIVATE "LINKER:--no-as-needed")

# Link all required libraries
target_link_libraries(codetr_inference
    argparse
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    TorchVision::TorchVision
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart
    ${TORCHTRT_LIB}
    ${TORCHTRT_RUNTIME_LIB}
    ${TORCHTRT_PLUGINS_LIB}
)

# Revert linker flag after TRTEngine registration
target_link_options(codetr_inference PRIVATE "LINKER:--as-needed")