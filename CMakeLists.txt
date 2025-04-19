cmake_minimum_required(VERSION 3.18)
project(codetr_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Searching for CUDAToolkit")
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDA libraries:    ${CUDAToolkit_LIBRARY_DIR}")
if(TARGET CUDA::cudart)
    get_target_property(CUDART_LIB CUDA::cudart IMPORTED_LOCATION)
    message(STATUS "CUDA::cudart resolves to: ${CUDART_LIB}")
else()
    message(WARNING "CUDA::cudart target does not exist!")
endif()


message(STATUS "Searching for OpenCV")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "Searching for Torch")
if (NOT DEFINED TORCH_CUDA_ARCH_LIST)
    message(WARNING "Defaulting to TORCH_CUDA_ARCH_LIST=8.9")
    set(TORCH_CUDA_ARCH_LIST "8.9")
endif()
message(STATUS "Setting TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
find_package(Torch REQUIRED)
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

message(STATUS "Searching for TorchVision")
find_package(TorchVision REQUIRED)
get_target_property(TARGET_TYPE TorchVision::TorchVision TYPE)
message(STATUS "TorchVision::TorchVision TYPE = ${TARGET_TYPE}")


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
    ${CUDAToolkit_INCLUDE_DIRS}     # for cuda_runtime_api.h
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${TORCHTRT_DIR}/include
    ${argparse_SOURCE_DIR}/include
)

# Add linker flags before Torch-TensorRT libs
target_link_options(codetr_inference PRIVATE "LINKER:--no-as-needed")

# Link all required libraries
target_link_libraries(codetr_inference PRIVATE
    ${TORCH_LIBRARIES}          # torch, c10, etc.
    TorchVision::TorchVision    # uses "torch"
    ${OpenCV_LIBS}               # cv::imread, etc.
    argparse
    CUDA::cudart
    ${TENSORRT_LIB}
    ${TENSORRT_PLUGIN_LIB}
    ${TORCHTRT_LIB}
    ${TORCHTRT_RUNTIME_LIB}
    ${TORCHTRT_PLUGINS_LIB}
)

target_link_options(codetr_inference PRIVATE "LINKER:--as-needed")
target_compile_options(codetr_inference PRIVATE -Wno-deprecated-declarations)