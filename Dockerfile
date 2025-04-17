FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel


# Update and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    cmake \
    build-essential \
    libopencv-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT 10.7.0.23 from tarball
ARG TRT_VERSION=10.7.0.23
ARG TRT_TARBALL=TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-12.6.tar.gz
ARG TRT_URL=https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/${TRT_TARBALL}
WORKDIR /opt
RUN wget --quiet --content-disposition ${TRT_URL} && \
    tar -xzf ${TRT_TARBALL} && \
    rm ${TRT_TARBALL}

# Set TensorRT environment variables
ENV TENSORRT_ROOT=/opt/TensorRT-${TRT_VERSION}
ENV LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:$LD_LIBRARY_PATH
ENV CPATH=${TENSORRT_ROOT}/include
ENV LIBRARY_PATH=${TENSORRT_ROOT}/lib:$LIBRARY_PATH
ENV PATH=${TENSORRT_ROOT}/bin:$PATH

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu126

# Clone and install MMDetection v3.3.0 as a regular package
RUN git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git /opt/mmdetection && \
    pip install --no-cache-dir -r /opt/mmdetection/requirements/mminstall.txt && \
    pip install --no-cache-dir /opt/mmdetection

# This ARG changes with each build to invalidate the following steps
ARG CACHE_BUSTER=manual

ENV PYTORCH_ROOT=/opt/conda/lib/python3.11/site-packages/torch
ENV LD_LIBRARY_PATH=${PYTORCH_ROOT}/lib:$LD_LIBRARY_PATH

# Optional: Set build-time GPU architecture
# for example, CUDA_ARCH="89;90"
# and TORCH_CUDA_ARCH_LIST="8.9;9.0"
ARG CUDA_ARCH="89"
ARG TORCH_CUDA_ARCH_LIST="8.9"

# Copy codetr source AFTER all cached layers (for faster rebuilds)
COPY . /workspace/codetr
# clean up the build directories
WORKDIR /workspace/codetr
RUN rm -rf build && rm -rf csrc_tests/build && rm -rf codetr/csrc/build && rm -f codetr/*.so

# Build the C++ plugin and copy the .so to the codetr directory
WORKDIR /workspace/codetr/codetr/csrc
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DTENSORRT_LIB_DIR="${TENSORRT_ROOT}/lib" \
    -DTENSORRT_INCLUDE_DIR="${TENSORRT_ROOT}/include" \
    && make -j$(nproc) \
    && cp libdeformable_attention_plugin.so ../../

# [TEST] Load the plugin in C++
WORKDIR /workspace/codetr/csrc_tests
RUN pip install cuda-python==12.6.2.post1
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev -Wno-deprecated-declarations \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_LIB_DIR="${TENSORRT_ROOT}/lib" \
    && make -j$(nproc) \
    && ./test_plugin /workspace/codetr/codetr/csrc/build/libdeformable_attention_plugin.so

ENV CUDA_ARCH="${CUDA_ARCH}"
WORKDIR /workspace/codetr
# Install codetr Python package
RUN pip install ninja
# Install package with -e flag to make testing easier
RUN pip install --no-build-isolation -e .


# WORKDIR /workspace/codetr
# RUN mkdir -p build && cd build && \
#     cmake .. -Wno-dev \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
#     && make -j$(nproc)

WORKDIR /workspace/codetr