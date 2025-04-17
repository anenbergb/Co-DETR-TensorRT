cmake_minimum_required(VERSION 3.18)
project(deformable_attention_plugin LANGUAGES CXX CUDA)

enable_language(CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMake handles the CUDA flags using this property
# For example, CMake will automatically pass the flags 
# -gencode=arch=compute_89,code=sm_89 to NVCC
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "Defaulting to CMAKE_CUDA_ARCHITECTURES=89")
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
if (NOT DEFINED TORCH_CUDA_ARCH_LIST)
    message(WARNING "Defaulting to TORCH_CUDA_ARCH_LIST=8.9")
    set(TORCH_CUDA_ARCH_LIST "8.9")
endif()

message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Setting TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

# Dependencies
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries:    ${CUDAToolkit_LIBRARY_DIR}")
find_package(Torch REQUIRED)

# TensorRT paths (can be overridden externally)
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu" CACHE PATH "TensorRT lib directory")
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "TensorRT include directory")

message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libs:    ${TENSORRT_LIB_DIR}")

find_library(TENSORRT_LIB nvinfer PATHS ${TENSORRT_LIB_DIR} REQUIRED)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin PATHS ${TENSORRT_LIB_DIR} REQUIRED)

add_library(deformable_attention_plugin SHARED
    deformable_attention_plugin.cpp
    ms_deform_attn.cu
)

target_link_options(deformable_attention_plugin PRIVATE "LINKER:--no-as-needed")

# Include paths
target_include_directories(deformable_attention_plugin PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(deformable_attention_plugin
    ${TORCH_LIBRARIES}
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    CUDA::cudart
)

target_link_options(deformable_attention_plugin PRIVATE "LINKER:--as-needed")