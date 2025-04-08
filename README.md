# Co-DETR-TensorRT
Co-DETR to TensorRT export


# Installation

## Installing pytorch/TensorRT
https://pytorch.org/TensorRT/getting_started/installation.html
```
pip install torch torch-tensorrt tensorrt
```





Building the C++ CUDA extension for multiscale deformable attention operator and installing the python codetr package.
```
pip install ninja
pip install --no-build-isolation -e .
```



References
- https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial
- https://github.com/pytorch/vision/tree/main/torchvision/csrc


# Tests

Consider added more controlled capture of warnings
https://docs.pytest.org/en/stable/how-to/capture-warnings.html
```
pytest tests/test_multi_scale_deformable_attention.py -s

pytest tests/test_export.py -s
```

# Export
```
python export.py \
--dtype float16 \
--optimization-level 3 \
--output /home/bryan/expr/co-detr/export/codetr_fp16 \
--height 768 --width 1152
```

# Dynamic shapes make the export take way too long
# dh = Dim("dh", min=480, max=1280)
# dw = Dim("dw", min=480, max=2048)
# dynamic_shapes = {
#     "x": (Dim.STATIC, Dim.STATIC, dh, dw)
# }

# C++ Executable

## Installation
* Depends on libtorch
* torchvision C++ (for NMS operator)
* torch_tensorrt

### Libtorch
https://pytorch.org/get-started/locally/

/home/bryan/src/libtorch

### Torchvision
Build torchvision 
https://github.com/pytorch/vision/tree/main/examples/cpp
```
mkdir build
cd build
export Torch_DIR=/home/bryan/src/libtorch/share/cmake/Torch
export CMAKE_PREFIX_PATH=$Torch_DIR
cmake ..
cmake --build . --parallel 8
cmake --install . -DCMAKE_INSTALL_PREFIX=/home/bryan/src/libtorchvision
```
### Torch-TensorRT
Download a pre-built release https://github.com/pytorch/TensorRT/releases


## Building the codetr_inference C++ executable
```
export Torch_DIR=/home/bryan/src/libtorch/share/cmake/Torch
export TorchVision_DIR=/home/bryan/src/libtorchvision/share/cmake/TorchVision
export TORCH_TENSORRT_ROOT=/home/bryan/src/torch_tensorrt


mkdir build
cd build
cmake .. -Wno-dev \
-DCMAKE_PREFIX_PATH="/home/bryan/src/libtorch;/home/bryan/src/libtorchvision" \
-DTORCH_TENSORRT_ROOT=/home/bryan/src/torch_tensorrt
make -j8

export LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:/home/bryan/src/libtorchvision/lib:/home/bryan/src/torch_tensorrt/lib:$LD_LIBRARY_PATH

./codetr_inference \s
--model /home/bryan/expr/co-detr/export/codetr_fp16/codetr.ts \
--input ../assets/demo.jpg \
--output /home/bryan/expr/co-detr/export/codetr_fp16/cpp_output.jpg
```



## Building the multiscale deformable attention TensorRT plugin

Built with TensorRT 10.9, so IPluginV3 and IPluginCreatorV3One were used.

sudo apt-get install python3-dev
```
cmake .. -Wno-dev \
-DCMAKE_PREFIX_PATH="/home/bryan/src/libtorch"

cmake .. -Wno-dev \
-DCMAKE_PREFIX_PATH="/home/bryan/anaconda3/envs/mmcv/lib/python3.12/site-packages/torch"

make -j8
```

Building the csrc_tests

```
mkdir build
cd build
cmake .. -Wno-dev
make -j8
LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:$LD_LIBRARY_PATH ./test_plugin
```
// /home/bryan/src/Co-DETR-TensorRT/codetr/csrc/build

You can verify the embedded RPATH with
```
> readelf -d ./test_plugin | grep runpath
0x000000000000001d (RUNPATH)            Library runpath: [/home/bryan/src/Co-DETR-TensorRT/codetr/csrc_tests/../csrc/build]
```

Pytest test plugin

```
# ensure that cuda python is installed
pip install cuda-python

LD_LIBRARY_PATH=/home/bryan/anaconda3/envs/mmcv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH pytest test_plugin.py -s
```




////

Large Scale Jittering Configuration
In the context of this DINO (DETR with Improved deNoising anchOr boxes) object detection model configuration, use_lsj=True enables Large Scale Jittering (LSJ) as a data augmentation technique.

Large Scale Jittering is an advanced augmentation strategy that randomly resizes training images within a much wider range than traditional resizing methods. While standard augmentation might resize images within a limited range (e.g., 0.8-1.2× original size), LSJ typically applies more aggressive scaling (often 0.1-2.0× original size).

This technique is particularly beneficial for:

Improving model robustness to scale variations
Enhancing detection performance on objects of varying sizes
Reducing overfitting by creating more diverse training examples
LSJ has become a standard technique in modern object detection frameworks, especially when training transformer-based models like DINO on the COCO dataset, as indicated by the filename of this configuration.



# Deferent from the DINO, we use the NMS.


TensorRT requires static output shapes so the score thresholding and nms have to be moved outside of the model and left as post-processing steps


The Pytorch model is compiled to TensorRT with fallback eneabled `require_full_compilation=False` because some operations, namely the multiscale_deformable_attention CUDA operator cannot be run in TensorRT.

PyTorch fallback means the compiled model becomes a hybrid execution engine where:
1. Supported operations run in TensorRT for acceleration
2. Unsupported operations run in the original PyTorch framework

How It Works
1. Graph Partitioning: During compilation, your model is analyzed and split into subgraphs:
* TensorRT-compatible subgraphs get compiled to optimized TensorRT engines
* Incompatible operations remain as PyTorch code

2. Runtime Flow:
```Input → TensorRT Subgraph → PyTorch Subgraph → TensorRT Subgraph → ... → Output```

3. Engine Transitions: Data moves between TensorRT and PyTorch execution contexts automatically

Drawbacks from hybrid execution
1. Each transition between PyTorch and TensorRT has overhead
2. Less optimization than a fully TensorRT-compiled model
3.  Some acceleration benefits might be reduced by frequent context switches


The result of Torch -> TensorRT compilation is a "hybrid execution engine" composed of an alternating pattern of TorchTensorRTModule and GraphModule. The 



# Implementing a TensorRT Custom Operator

* https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
* https://github.com/leimao/TensorRT-Custom-Plugin-Example
* https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/custom_kernel_plugins.html
* https://pytorch.org/TensorRT/contributors/dynamo_converters.html
* https://github.com/NVIDIA/TensorRT/tree/release/10.0/samples/python/python_plugin

Example plugins
* https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins


There are four steps to ensure that TensorRT properly recognizes your plugin:

1. Implement a plugin class from one of TensorRT’s plugin base classes. Currently, the only recommended one is IPluginV3.
2. Implement a plugin creator class tied to your class by deriving from one of TensorRT’s plugin creator-based classes. Currently, the only recommended one is IPluginCreatorV3One.
3. Register an instance of the plugin creator class with TensorRT’s plugin registry.
4. Add an instance of the plugin class to a TensorRT network by directly using TensorRT’s network APIs



https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
* Very useful guide providing examples for IPluginV3 adn IPluginV3OneBuild
https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html
* TensorRT python API documentation


# Writing Dynamo Converters
https://pytorch.org/TensorRT/contributors/dynamo_converters.html
* unfortunately didn't provide a fulle xample
*  torch_tensorrt dynamo/conversion library https://github.com/pytorch/TensorRT/tree/v2.6.0/py/torch_tensorrt/dynamo/conversion
