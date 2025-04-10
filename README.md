# Co-DETR to TensorRT
This repository presents a refactored implementation of the [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR) Co-DETR object detection neural network architecture to enable export and compilation from [Pytorch](https://pytorch.org/) to [NVIDIA's TensorRT Deep Learning Optimization and Runtime framework](https://developer.nvidia.com/tensorrt).

Compiliation from Pytorch to TensorRT is accomplished in two steps
1. Trace the Co-DETR neural network graph via [`torch.export.export`](https://pytorch.org/docs/stable/export.html#torch.export.export) to yield an Export Intermediate Representation (IR) (`torch.export.ExportedProgram`) that bundles the computational graph (`torch.fx.GraphModule`), the graph signature specifying the parameters and buffer names used and mutated within the graph, and parameters and weights of the model. The Ahead-of-Time (AOT) tracing process removes all Python control flow and data structures, and recasts the computation as a series of [ATen operators](https://pytorch.org/executorch/stable/ir-ops-set-definition.html)

* References:
  * https://pytorch.org/docs/stable/export.ir_spec.html
  * https://pytorch.org/cppdocs/
2. Compile the traced IR graph to TensorRT via [`torch_tensorrt.dynamo.compile`](https://pytorch.org/TensorRT/py_api/dynamo.html). The [torch_tensorrt](https://pytorch.org/TensorRT/) library uses the [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) compiler to replace Pytorch operators in the traced graph with equivalent TensorRT operators. Operators unsupported in TensorRT will be left as Pytorch ATen operators, resulting a hybrid graph composed on ATen and TensorRT subgraphs. The TensorRT-compatible subgraphs are optimized and executed using TensorRT, while the remaining parts are handled by PyTorch's native execution. In general, a model fully compiled to TensorRT operators is expected to achieve better performance. To enable full TensorRT compilation of Co-DETR, I implemented a [C++ TensorRT Plugin for the Multi-Scale Deformable Attention operator](codetr/csrc/deformable_attention_plugin.cpp) and a [dynamo_tensorrt_converter wrapper](codetr/csrc/deformable_attention_plugin.cpp). The compiled Co-DETR TensorRT can be run in Python or C++ with the TorchScript frontend or with native TensorRT. To execute with native TensorRT, the model can be serialized to a TensorRT engine via `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine`.

* References
  * https://github.com/pytorch/TensorRT
  * https://pytorch.org/docs/stable/torch.compiler.html
  * https://pytorch.org/TensorRT/contributors/dynamo_converters.html


2.12x inference runtime speed-up is achieved 

|   Model         | Input Size (W,H) | Pytorch FP32 | Pytorch FP16 | TensorRT FP16 | Speed-up |
| :-------:       | :--------------: | :----------: | :----------: | :-----------: | :------: |
|  Co-DINO Swin-L | (1152, 768)      |  100 ms      | 50.77 ms     |  36.56ms      | 2.12x    |


# CO-DETR

> [DETRs with Collaborative Hybrid Assignments Training](https://arxiv.org/abs/2211.12860)

## Abstract

In this paper, we provide the observation that too few queries assigned as positive samples in DETR with one-to-one set matching leads to sparse supervision on the encoder's output which considerably hurt the discriminative feature learning of the encoder and vice visa for attention learning in the decoder. To alleviate this, we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. This new training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training the multiple parallel auxiliary heads supervised by one-to-many label assignments such as ATSS and Faster RCNN. In addition, we conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve the training efficiency of positive samples in the decoder. In inference, these auxiliary heads are discarded and thus our method introduces no additional parameters and computational cost to the original detector while requiring no hand-crafted non-maximum suppression (NMS). We conduct extensive experiments to evaluate the effectiveness of the proposed approach on DETR variants, including DAB-DETR, Deformable-DETR, and DINO-Deformable-DETR. The state-of-the-art DINO-Deformable-DETR with Swin-L can be improved from 58.5% to 59.5% AP on COCO val. Surprisingly, incorporated with ViT-L backbone, we achieve 66.0% AP on COCO test-dev and 67.9% AP on LVIS val, outperforming previous methods by clear margins with much fewer model sizes.

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/dceaf7ee-cd6c-4be0-b7b1-5b01a7f11724"/>
</div>

## Results and Models

|   Model   | Backbone | Epochs | Aug  |            Dataset            | box AP |                                 Config                                 |                                                                                                                                                     Download                                                                                                                                                      |
| :-------: | :------: | :----: | :--: | :---------------------------: | :----: | :--------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Co-DINO  |   R50    |   12   | LSJ  |             COCO              |  52.0  |    [config](configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py)     | [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_lsj_8xb2_1x_coco/co_dino_5scale_r50_lsj_8xb2_1x_coco-69a72d67.pth)\\ [log](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_lsj_8xb2_1x_coco/co_dino_5scale_r50_lsj_8xb2_1x_coco_20230818_150457.json) |
| Co-DINO\* |   R50    |   12   | DETR |             COCO              |  52.1  |      [config](configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py)       |                                                                                                      [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_1x_coco-7481f903.pth)                                                                                                      |
| Co-DINO\* |   R50    |   36   | LSJ  |             COCO              |  54.8  |    [config](configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py)     |                                                                                                    [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth)                                                                                                    |
| Co-DINO\* |  Swin-L  |   12   | DETR |             COCO              |  58.9  |    [config](configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py)     |                                                                                                  [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_1x_coco-27c13da4.pth)                                                                                                   |
| Co-DINO\* |  Swin-L  |   12   | LSJ  |             COCO              |  59.3  |  [config](configs/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py)   |                                                                                                [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth)                                                                                                 |
| Co-DINO\* |  Swin-L  |   36   | DETR |             COCO              |  60.0  |    [config](configs/codino/co_dino_5scale_swin_l_16xb1_3x_coco.py)     |                                                                                                  [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_3x_coco-d7a6d8af.pth)                                                                                                   |
| Co-DINO\* |  Swin-L  |   36   | LSJ  |             COCO              |  60.7  |  [config](configs/codino/co_dino_5scale_swin_l_lsj_16xb1_3x_coco.py)   |                                                                                                [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_lsj_swin_large_1x_coco-3af73af2.pth)                                                                                                 |
| Co-DINO\* |  Swin-L  |   16   | DETR | Objects365 pre-trained + COCO |  64.1  | [config](configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py) |                                                                                               [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth)                                                                                               |

Note

- Models labeled * are not trained by us, but from [CO-DETR](https://github.com/Sense-X/Co-DETR) official website.
- We find that the performance is unstable and may fluctuate by about 0.3 mAP.
- If you want to save GPU memory by enabling checkpointing, please use the `pip install fairscale` command.



# Installation

## Installing TensorRT
The version of TensorRT used to build and run the C++ Co-DETR inference executable must be the same version as that used to compile the Co-DETR model engine in python.

This may require downloading a previous version of TensorRT
https://developer.nvidia.com/nvidia-tensorrt-download


### pytorch/TensorRT
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
mkdir build
cd build
cmake .. -Wno-dev \
-DCMAKE_PREFIX_PATH="/home/bryan/src/libtorch;/home/bryan/src/libtorchvision" \
-DTORCH_TENSORRT_ROOT=/home/bryan/src/torch_tensorrt \
-DTENSORRT_DIR=/home/bryan/src/TensorRT-10.7.0.23
make -j8

LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:/home/bryan/src/libtorchvision/lib:/home/bryan/src/torch_tensorrt/lib:/home/bryan/src/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH ./codetr_inference \
--model /home/bryan/expr/co-detr/export/codetr_fp16/codetr.ts \
--input ../assets/demo.jpg \
--output /home/bryan/expr/co-detr/export/codetr_fp16/cpp_ts_output.jpg \
--benchmark-iterations 100


LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:/home/bryan/src/libtorchvision/lib:/home/bryan/src/torch_tensorrt/lib:/home/bryan/src/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH ./codetr_inference \
--model /home/bryan/expr/co-detr/export/codetr_fp16/codetr.engine \
--input ../assets/demo.jpg \
--output /home/bryan/expr/co-detr/export/codetr_fp16/cpp_engine_output.jpg \
--benchmark-iterations 100

```

Then to show that TensorRT is linked correctly
```
> ldd codetr_inference | grep nvinfer
        libnvinfer.so.10 => /home/bryan/src/TensorRT-10.7.0.23/lib/libnvinfer.so.10 (0x00007ca307600000)
        libnvinfer_plugin.so.10 => /home/bryan/src/TensorRT-10.7.0.23/lib/libnvinfer_plugin.so.10 (0x00007ca305200000)
```



### ABI  Mismatch

Make sure the TorchVision and Torch-TensorRT libraries you're linking to are built with the same Torch version and ABI.
If Torch-TensorRT or TorchVision was built from source:
* Verify they're using the same PyTorch headers/libraries you’re linking against.
* Check their ABI settings (they must match your -D_GLIBCXX_USE_CXX11_ABI=1).


nm -C /home/bryan/src/libtorch/lib/libtorch.so | grep 'std::__cxx11'  -> no results
nm -C /home/bryan/src/libtorchvision/lib/libtorchvision.so | grep 'std::__cxx11' -> lots of results
nm -C /home/bryan/src/torch_tensorrt/lib/libtorchtrt.so | grep 'std::__cxx11' -> lots of results

If you see lots of std::__cxx11::string, it's using the new ABI.

torch 2.6

nm -C /home/bryan/anaconda3/envs/mmcv/lib/python3.12/site-packages/torch_tensorrt/lib/libtorchtrt.so | grep 'std::string' | grep '__cxx11'


// Ensure that TensorRT versions are the same. The version used in Python, and the version installed on device
python:

tensorrt.__version__
'10.7.0.post1

>>> import tensorrt
>>> tensorrt.__version__
'10.9.0.34'


> nm -D /usr/lib/x86_64-linux-gnu/libnvinfer.so | grep tensorrt
00000000224b0524 B tensorrt_build_root_20250301_9406843d50261530
00000000224b0528 B tensorrt_version_10_9_0_34

cat /usr/include/x86_64-linux-gnu/NvInferVersion.h




## Building the multiscale deformable attention TensorRT plugin

Built with TensorRT 10.9, so IPluginV3 and IPluginCreatorV3One were used.

sudo apt-get install python3-dev
```

cmake .. -Wno-dev \
-DCMAKE_PREFIX_PATH="/home/bryan/anaconda3/envs/mmcv/lib/python3.12/site-packages/torch" \
-DTENSORRT_DIR=/home/bryan/src/TensorRT-10.7.0.23
make -j8
```
// verify the version of TensorRT that we are linking against\
```
> ldd libdeformable_attention_plugin.so | grep nvinfer
    libnvinfer.so.10 => /home/bryan/src/TensorRT-10.7.0.23/lib/libnvinfer.so.10 (0x000079efcf800000)
```

Building the csrc_tests

```
mkdir build
cd build
cmake .. -Wno-dev \
-DTENSORRT_DIR=/home/bryan/src/TensorRT-10.7.0.23

make -j8
LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:/home/bryan/src/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH ./test_plugin
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

TensorRT C++ Documentation
* https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/index.html
* https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html


# Writing Dynamo Converters
https://pytorch.org/TensorRT/contributors/dynamo_converters.html
* unfortunately didn't provide a fulle xample
*  torch_tensorrt dynamo/conversion library https://github.com/pytorch/TensorRT/tree/v2.6.0/py/torch_tensorrt/dynamo/conversion


# Inference time speed-up

Not including the post-processing code
```
PyTorch implementation:
  50.77 ms
  1 measurement, 10 runs , 1 thread
TensorRT implementation:
  36.56 ms
  1 measurement, 100 runs , 1 thread
TensorRT speedup: 1.39x 
```

C++
```
TorchScript: 
Average inference time: 36.46ms
TensorRT:
Average inference time: 36.50ms
```
