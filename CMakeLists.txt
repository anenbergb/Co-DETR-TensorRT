cmake_minimum_required(VERSION 3.18)
project(codetr_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TorchVision REQUIRED)

# Add argparse library
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

# Set torch_tensorrt paths (update these paths according to your installation)
if(NOT DEFINED TORCH_TENSORRT_ROOT)
    message(FATAL_ERROR "TORCH_TENSORRT_ROOT not specified. Please provide it with -DTORCH_TENSORRT_ROOT=/path/to/torch_tensorrt")
endif()

set(TORCH_TENSORRT_INCLUDE_DIRS "${TORCH_TENSORRT_ROOT}/include")
set(TORCH_TENSORRT_LIBRARIES "${TORCH_TENSORRT_ROOT}/lib")

# Add executable
add_executable(codetr_inference codetr_inference.cpp)

# Link libraries
target_link_libraries(codetr_inference
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    argparse
    TorchVision::TorchVision
    ${TORCH_TENSORRT_LIBRARIES}/libtorchtrt.so
    ${TORCH_TENSORRT_LIBRARIES}/libtorchtrt_runtime.so
    ${TORCH_TENSORRT_LIBRARIES}/libtorchtrt_plugins.so
)

# Include directories
target_include_directories(codetr_inference PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${argparse_SOURCE_DIR}/include
    ${TORCHVISION_INCLUDE_DIRS}
    ${TORCH_TENSORRT_INCLUDE_DIRS}
)

# Set CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_89 -code=sm_89") 