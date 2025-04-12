FROM nvcr.io/nvidia/pytorch:24.12-py3

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Clone MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git
WORKDIR /workspace/mmdetection

# Install MMEngine and MMCV dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements/mminstall.txt \
 && pip install .

# Optional: Install additional packages commonly used
RUN pip install opencv-python-headless matplotlib pycocotools

# Done
WORKDIR /workspace/mmdetection
