cmake_minimum_required(VERSION 3.18)
project(deformable_attention_plugin LANGUAGES CXX CUDA)

enable_language(CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

# Dependencies
find_package(Torch REQUIRED)

add_library(deformable_attention_plugin SHARED
    deformable_attention_plugin.cpp
    ms_deform_attn.cu
)

target_link_options(deformable_attention_plugin PRIVATE "LINKER:--no-as-needed")

# Include paths
target_include_directories(deformable_attention_plugin PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(deformable_attention_plugin
    ${TORCH_LIBRARIES}
    nvinfer
    nvinfer_plugin
    cudart
)

target_link_options(deformable_attention_plugin PRIVATE "LINKER:--as-needed")

# CUDA Architectures (e.g. for Ada/Hopper GPUs, sm_89)
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set_target_properties(deformable_attention_plugin PROPERTIES
        CUDA_ARCHITECTURES 89
    )
endif()
