FROM nvcr.io/nvidia/pytorch:24.12-py3

# Optional: Set build-time GPU architecture
# for example, CUDA_ARCH="89;90"
# and TORCH_CUDA_ARCH_LIST="8.9;9.0"
ARG CUDA_ARCH="89"
ARG TORCH_CUDA_ARCH_LIST="8.9"

# Update and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
# Clone and install MMDetection v3.3.0 as a regular package
RUN git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git /mmdetection && \
    pip install --no-cache-dir -r /mmdetection/requirements/mminstall.txt && \
    pip install --no-cache-dir /mmdetection

# This ARG changes with each build to invalidate the following steps
ARG CACHE_BUSTER=manual

# Copy codetr source AFTER all cached layers (for faster rebuilds)
COPY . /workspace/codetr

# Build the C++ plugin and copy the .so to the codetr directory
WORKDIR /workspace/codetr/codetr/csrc
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && make -j$(nproc) \
    && cp libdeformable_attention_plugin.so ../../

WORKDIR /workspace/codetr/codetr/csrc_tests
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

ENV CUDA_ARCH="${CUDA_ARCH}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

WORKDIR /workspace/codetr
# Install codetr Python package
RUN pip install ninja
# Install package with -e flag to make testing easier
RUN pip install --no-build-isolation -e .





# WORKDIR /workspace/codetr
# RUN mkdir -p build && cd build && \
#     cmake .. -Wno-dev \
#     -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
#     && make -j$(nproc)

WORKDIR /workspace