# Local Installation
To install the library locally, you'll need a similar environment to the [Dockerfile](Dockerfile).

The library was tested with
* CUDA 12.6
* TensorRT 10.7.0.23
* PyTorch 2.6.0
* Torch-TensorRT 2.6.0

## 1. Install CUDA version 12.6
Download and install the CUDA library locally https://developer.nvidia.com/cuda-12-6-0-download-archive

## 2. Download TensorRT version 10.7.0.23
Download TensorRT locally https://developer.nvidia.com/nvidia-tensorrt-download
```
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-12.6.tar.gz
```
Install to a directory, e.g. `TensorRT_DIR="/home/bryan/src/TensorRT-10.7.0.23"`


## 3. Follow Dockerfile build example

Set the CUDA_ARCH to your GPU Compute Compatibilty https://developer.nvidia.com/cuda-gpus

```
conda create -n codetr python=3.12 -y
conda activate codetr

pip install --upgrade pip setuptools wheel
pip install torch torchvision torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/cu126
git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git /home/bryan/Downloads/mmdetection && \
pip install --no-cache-dir -r /home/bryan/Downloads/mmdetection/requirements/mminstall.txt && \
pip install --no-cache-dir /home/bryan/Downloads/mmdetection

CUDA_ARCH="89"
TORCH_CUDA_ARCH_LIST="8.9"
TensorRT_DIR="/home/bryan/src/TensorRT-10.7.0.23"
DIR=$(pwd)

cd $DIR/codetr/csrc
mkdir -p build && cd build
cmake .. -Wno-dev \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
-DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
-DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
-DTENSORRT_LIB_DIR="${TensorRT_DIR}/lib" \
-DTENSORRT_INCLUDE_DIR="${TensorRT_DIR}/include"
make -j$(nproc)
cp libdeformable_attention_plugin.so ../../

# [TEST] Load the plugin in C++
cd $DIR/csrc_tests
mkdir -p build && cd build
cmake .. -Wno-dev \
-DCMAKE_BUILD_TYPE=Release \
-DTENSORRT_LIB_DIR="${TensorRT_DIR}/lib"
make -j$(nproc)
LD_LIBRARY_PATH="${TensorRT_DIR}/lib":$LD_LIBRARY_PATH ./test_plugin $DIR/codetr/csrc/build/libdeformable_attention_plugin.so

# [TEST] Run the plugin in python
cd $DIR/csrc_tests
pip install cuda-python pytest
LD_LIBRARY_PATH="${TensorRT_DIR}/lib":$LD_LIBRARY_PATH pytest test_plugin.py -s --plugin-lib $DIR/codetr/csrc/build/libdeformable_attention_plugin.so

cd $DIR
pip install ninja
export CUDA_ARCH=$CUDA_ARCH
pip install --no-build-isolation -e .

# [TEST] test torch -> tensorrt compilation
LD_LIBRARY_PATH="${TensorRT_DIR}/lib":$LD_LIBRARY_PATH pytest tests/test_multi_scale_deformable_attention.py -s
LD_LIBRARY_PATH="${TensorRT_DIR}/lib":$LD_LIBRARY_PATH pytest tests/test_export.py -s
```

## 4. Export the Co-DETR model from Pytorch to TensorRT

Download a copy of CO-DETR weights [co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth) 

```
python export.py \
--dtype float16 \
--optimization-level 3 \
--output output_fp16 \
--height 768 --width 1152 \
--weights co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
--plugin-lib codetr/csrc/build/libdeformable_attention_plugin.so
```

## 5. Download LibTorch, TorchVision, and Torch-TensorRT C++ libraries
You can run the compiled TorchScript or TensorRT serialized engine in C++ without dependency on python libraries,
as long as the library version matches with the Python library used to compile the model.

### Download LibTorch 2.6.0 for CUDA 12.6
Download LibTorch locally https://pytorch.org/get-started/locally/
```
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip
```
Install to a directory, e.g. `/home/bryan/src/libtorch`

### Download TorchVision
Torchvision C++ is required for the NMS operator

Clone [torchvision github](https://github.com/pytorch/vision/tree/main) and follow the [C++ installation instructions](https://github.com/pytorch/vision/tree/main/examples/cpp) to build locally 
```
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/bryan/src/libtorch/share/cmake/Torch
cmake --build . --parallel 8
cmake --install . -DCMAKE_INSTALL_PREFIX=/home/bryan/src/libtorchvision
```
Install to a directory, e.g. `/home/bryan/src/libtorchvision`

### Download Torch-TensorRT 2.6.0
Download Torch-TensorRT pre-compiled library. This must be the same version that's installed in python. See [TensorRT/release v2.6.0](https://github.com/pytorch/TensorRT/releases)
```
wget https://github.com/pytorch/TensorRT/releases/download/v2.6.0/libtorchtrt-2.6.0-tensorrt10.7.0.post1-cuda124-libtorch2.6.0-x86_64-linux.tar.gz
```
Install to a directory, e.g. `/home/bryan/src/torch_tensorrt`

### 6. Build and run C++ executable

Build the codetr_inference executable
```
mkdir -p build && cd build
cmake .. -Wno-dev \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="/home/bryan/src/libtorch;/home/bryan/src/libtorchvision" \
-DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
-DTENSORRT_LIB_DIR="${TensorRT_DIR}/lib" \
-DTENSORRT_INCLUDE_DIR="${TensorRT_DIR}/include" \
-DTORCHTRT_DIR=/home/bryan/src/torch_tensorrt
make -j$(nproc)
```

Run inference C++ using the compiled TorchScript model 
```
LD_LIBRARY_PATH="/home/bryan/src/libtorch/lib:/home/bryan/src/libtorchvision/lib:/home/bryan/src/torch_tensorrt:${TensorRT_DIR}/lib:$LD_LIBRARY_PATH" \
./codetr_inference \
--model ../output_fp16/codetr.ts \
--input ../assets/demo.jpg \
--output ../output_fp16/cpp_ts_output.jpg \
--benchmark-iterations 100 \
--trt-plugin-path ../codetr/csrc/build/libdeformable_attention_plugin.so \
--dtype float16 --target-height 768 --target-width 1152
```

Run inference in C++ using the compile TensorRT serialized engine file
```
LD_LIBRARY_PATH="/home/bryan/src/libtorch/lib:/home/bryan/src/libtorchvision/lib:/home/bryan/src/torch_tensorrt:${TensorRT_DIR}/lib:$LD_LIBRARY_PATH" \
./codetr_inference \
--model ../output_fp16/codetr.engine \
--input ../assets/demo.jpg \
--output ../output_fp16/cpp_engine_output.jpg \
--benchmark-iterations 100 \
--trt-plugin-path ../codetr/csrc/build/libdeformable_attention_plugin.so \
--dtype float16 --target-height 768 --target-width 1152
```