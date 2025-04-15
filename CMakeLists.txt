cmake_minimum_required(VERSION 3.18)
project(codetr_inference)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries:    ${CUDAToolkit_LIBRARY_DIR}")

# Allow custom TensorRT paths, fallback to system defaults
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT lib directory")
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include directory")

message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libs:    ${TENSORRT_LIB_DIR}")

find_library(TENSORRT_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} REQUIRED)


# Allow user to pass TORCH_TENSORRT_DIR, or use default
set(TORCH_TENSORRT_DIR "/usr/local/lib/python3.12/dist-packages/torch_tensorrt" CACHE PATH "Path to Torch-TensorRT installation")
message(STATUS "Using Torch-TensorRT from: ${TORCH_TENSORRT_DIR}")

set(TORCH_TENSORRT_INCLUDE_DIRS "${TORCH_TENSORRT_DIR}/include")
set(TORCH_TENSORRT_LIBRARY_DIR "${TORCH_TENSORRT_DIR}/lib")

# Add the executable
add_executable(codetr_inference codetr_inference.cpp)

# Include paths
target_include_directories(codetr_inference PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${argparse_SOURCE_DIR}/include
    ${TORCHVISION_INCLUDE_DIRS}
    ${TORCH_TENSORRT_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# Find Torch-TensorRT shared libraries
find_library(TORCHTRT_LIB torchtrt PATHS ${TORCH_TENSORRT_LIBRARY_DIR} REQUIRED)
find_library(TORCHTRT_RUNTIME_LIB torchtrt_runtime PATHS ${TORCH_TENSORRT_LIBRARY_DIR} REQUIRED)
find_library(TORCHTRT_PLUGINS_LIB torchtrt_plugins PATHS ${TORCH_TENSORRT_LIBRARY_DIR} REQUIRED)

# Add linker flags before Torch-TensorRT libs
target_link_options(codetr_inference PRIVATE "LINKER:--no-as-needed")

# Link all required libraries
target_link_libraries(codetr_inference
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    TorchVision::TorchVision
    argparse
    ${TORCHTRT_LIB}
    ${TORCHTRT_RUNTIME_LIB}
    ${TORCHTRT_PLUGINS_LIB}
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart
)

# Revert linker flag after TRTEngine registration
target_link_options(codetr_inference PRIVATE "LINKER:--as-needed")

# CUDA config (if required by your model or dependencies)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_89 -code=sm_89")
