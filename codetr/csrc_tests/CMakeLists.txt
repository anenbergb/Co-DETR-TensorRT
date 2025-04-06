cmake_minimum_required(VERSION 3.18)
project(test_deformable_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_plugin test_plugin.cpp)

# Link to your plugin .so and the core TRT libraries
#  - If your plugin library is in a different folder, point to it accordingly
#  - You may also need to link to cudart if you haven't already
target_link_libraries(test_plugin
    # deformable_attention_plugin   # <--- your plugin .so
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc/build/libdeformable_attention_plugin.so
    nvinfer
    nvinfer_plugin
    cudart
)
