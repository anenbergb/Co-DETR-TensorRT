FROM nvcr.io/nvidia/pytorch:24.12-py3

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

# Copy codetr source AFTER all cached layers (for faster rebuilds)
COPY . /workspace/codetr
WORKDIR /workspace/codetr

# Install codetr Python package
RUN pip install ninja
RUN pip install --no-build-isolation -e .

# Build the C++ plugin
WORKDIR /workspace/codetr/codetr/csrc
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && make -j$(nproc)

WORKDIR /workspace/codetr/codetr/csrc_tests
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    && make -j$(nproc)

WORKDIR /workspace/codetr
RUN mkdir -p build && cd build && \
    cmake .. -Wno-dev \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    && make -j$(nproc)

WORKDIR /workspace