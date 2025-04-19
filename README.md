# Co-DETR (Detection Transformer) compiled from PyTorch to TensorRT
This repository presents a refactored implementation of the [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR) Co-DETR object detection neural network architecture to enable export and compilation from [Pytorch](https://pytorch.org/) to [NVIDIA's TensorRT Deep Learning Optimization and Runtime framework](https://developer.nvidia.com/tensorrt).

Compiliation from Pytorch to TensorRT is accomplished in two steps
### 1. Ahead-of-Time (AOT) graph tracing:
The Co-DETR neural network graph is traced via [`torch.export.export`](https://pytorch.org/docs/stable/export.html#torch.export.export) to yield an Export Intermediate Representation (IR) (`torch.export.ExportedProgram`) that bundles the computational graph (`torch.fx.GraphModule`), the graph signature specifying the parameters and buffer names used and mutated within the graph, and the parameters and weights of the model. The tracing process removes all Python control flow and data structures, and recasts the computation as a series of functional [ATen operators](https://pytorch.org/executorch/stable/ir-ops-set-definition.html).


References:
- https://pytorch.org/docs/stable/export.ir_spec.html
- https://pytorch.org/cppdocs/

### 2. TorchDynamo compilation to TensorRT:
The traced IR graph is compiled to TensorRT via [`torch_tensorrt.dynamo.compile`](https://pytorch.org/TensorRT/py_api/dynamo.html). The [torch_tensorrt](https://pytorch.org/TensorRT/) library uses the [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) compiler to replace Pytorch ATen operators in the traced graph with equivalent TensorRT operators. Operators unsupported in TensorRT are left as Pytorch ATen operators, resulting a hybrid graph composed on Pytorch ATen and TensorRT subgraphs. The TensorRT-compatible subgraphs are optimized and executed using TensorRT, while the remaining parts are handled by PyTorch's native execution. In general, a model fully compiled to TensorRT operators will achieve better performance. To enable full TensorRT compilation of Co-DETR, I implemented a [C++ TensorRT Plugin for the Multi-Scale Deformable Attention operator](codetr/csrc/deformable_attention_plugin.cpp) and a [dynamo_tensorrt_converter wrapper](codetr/csrc/deformable_attention_plugin.cpp). 

The compiled Co-DETR TensorRT model can be saved to disk as a [TorchScript](https://pytorch.org/docs/stable/jit.html) model via [`torch_tensorrt.save`](https://pytorch.org/TensorRT/py_api/torch_tensorrt.html?highlight=save#torch_tensorrt.save). The TorchScript can be standalone executed in Python or C++.

Furthermore, the Co-DETR TensorRT model can be serialized to a TensorRT engine via `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine` and saved to disk as a binary file. The serialized TensorRT engine can be run natively with TensorRT in Python or C++. 


References
* https://github.com/pytorch/TensorRT
* https://pytorch.org/docs/stable/torch.compiler.html
* https://pytorch.org/TensorRT/contributors/dynamo_converters.html


### 4x Inference Runtime Improvement (relative to Pytorch FP32) when executing the TensorRT float16 compiled Co-DETR model

All benchmarking is performed on an Nvidia RTX 4090 GPU.

|   Model                                         | Input Size (W,H) | Pytorch FP32 | Pytorch FP16 | TensorRT FP16 | Speed-up | 
| :-------:                                       | :--------------: | :----------: | :----------: | :-----------: | :------: |
|  Co-DINO Swin-L (Objects365 pre-trained + COCO) | (1920, 1280)     |  346 ms      |  147.1 ms    |  79.5 ms      | 4.35x    |
|  Co-DINO Swin-L (Objects365 pre-trained + COCO) | (1152, 768)      |  123 ms      |  50.8 ms     |  30.2ms       | 4.07x    |
|  Co-DINO Swin-L (Objects365 pre-trained + COCO) | (608, 608)       |  59 ms       |  24.8 ms     |  13.4ms       | 4.40x    |

The TensorRT FP16 runtimes are the mean GPU compute times reported using `trtexec` with 100 iterations and `--useSpinWait --useCudaGraph` options.
I recorded slightly longer runtimes of 99.1ms, 36.6ms, and 16.8ms when benchmarking with the `codetr_inference.cpp` script.
Check out the "Benchmarking with trtexec" section in this README for more information on inference runtime benchmarking.

Note that the Swin-L backbone downscales by a factor of 32x

The Co-DINO Swin-L model used here was pulled from the [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR) results table

|   Model   | Backbone | Epochs | Aug  | Dataset | box AP   |Config | Download |
| :-------: | :------: | :----: | :--: | :------: | :-----: | :----: | :---: |
| Co-DINO\* |  Swin-L  |   16   | DETR | Objects365 pre-trained + COCO |  64.1  | [config](configs/codino/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py) |                                                                                               [model](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth)   


# CO-DETR: Collaborative Detection Transformer
> Original publication: DETRs with Collaborative Hybrid Assignments Training https://arxiv.org/abs/2211.12860
<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/dceaf7ee-cd6c-4be0-b7b1-5b01a7f11724"/>
</div>

Co-DETR ranks at the top of the [object detection leaderboard](https://paperswithcode.com/sota/object-detection-on-coco) with a box mAP score of `66.0` on the COCO test-dev dataset.

Co-DETR (Collaborative-DETR) is an advanced object detector that builds upon the DETR family by introducing a collaborative hybrid assignment strategy to improve training efficiency and detection accuracy.
* One-to-one assignment (as in DETR) allows each ground-truth box to be matched with exactly one query via the Hungarian algorithm.
* One-to-many assignment introduces auxiliary losses where multiple queries can match a single ground-truth box, improving optimization and convergence.
* This hybrid assignment leads to better learning signals and faster convergence, especially in the early stages of training.
* The collaborative hybrid assignment strategy is compatible with existing DETR variants including Deformable-DETR and DINO-Deformable-DETR.

### Comparing DETR variants

| **Component**                  | **DETR (2020)**                                           | **Deformable DETR (2021)**                                        | **DINO (2022)**                                                      | **Co-DETR (2022)**                                                    |
|-------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|
| **Backbone**                  | ResNet                                                    | ResNet                                                     | ResNet / Swin / ViT                                                 | ResNet / Swin / ViT                                               |
| **Encoder Attention**         | Full global attention                                     | Sparse deformable attention                                       | Sparse deformable attention                                         | Sparse deformable attention (optional)                                |
| **Multi-Scale Features**      | ❌ Only last FPN level                                    | ✔️ Multi-scale deformable attention                                | ✔️ Multi-scale deformable attention                                 | ✔️ Multi-scale support (Deformable attention or FPN)                   |
| **Positional Encoding**       | Fixed sine-cosine                                         | Learned reference points                                          | Improved sine + learned reference points                            | Follows DINO / Deformable (flexible)                                  |
| **Query Design**              | Fixed learnable queries                                   | Queries with learnable reference points                           | **Dynamic Anchor Boxes (DAB)** with iterative refinement            | Learnable queries + hybrid one-to-one/one-to-many assignments         |
| **Matching Strategy**         | One-to-one (Hungarian)                                    | One-to-one (Hungarian)                                            | One-to-one + **one-to-many (auxiliary)**                           | One-to-one + **one-to-many (hybrid branch)**                          |
| **Auxiliary Losses**          | Intermediate decoder layers only                          | Used                                                               | Used + **Denoising Training (DN)**                                  | One-to-many branch used to boost one-to-one learning                  |
| **Two-Stage Detection**       | ❌                                                         | ✔️ Optional RPN-like region proposal + refinement                  | ✔️ Commonly used in DINO implementations                            | ✔️ Supported; used in most Co-DETR variants                           |
| **Training Stability**        | Sensitive                                                 | Improved                                                           | Robust due to DN and dynamic anchors                                | Very stable due to collaborative assignment                           |
| **Convergence Speed**         | ❌ Slow (500 epochs)                                      | ✅ Fast (~50 epochs)                                               | ✅✅ Very fast (12–36 epochs)                                        | ✅✅ Fast (12–24 epochs), faster than DINO in some variants            |
| **Key Innovation**            | End-to-end transformer detection                          | Deformable multi-scale sparse attention                           | Denoising + Dynamic Anchors + One-to-many supervision               | **Hybrid one-to-one & one-to-many assignment for collaborative optimization** |
| **Performance (COCO mAP) test-dev**    | ~42.0 (Resnet50, 500 epochs)                        | ~50.0 (Resnet50, 50 epochs)                         | ~63.3 (Swin-L, 36 epochs)                 | ~64.1 (Swin-L, 16 e99.11ms


# Installation (Docker Container)
The easiest way to install the library and reproduce the results is to build the provided [Dockerfile](Dockerfile). If you would prefer to install locally, see [INSTALL_LOCAL.md](INSTALL_LOCAL.md).


### 1. Install Prerequisites
* NVIDIA GPU drivers installed
* Docker installed: https://docs.docker.com/get-docker
* NVIDIA Container Toolkit
```
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Build the Dockerfile
The [Dockerfile](Dockerfile) is built from the [Pytorch docker base image pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags)

The environment includes
* Ubuntu 24.04 
* CUDA 12.6.3
* PyTorch 2.6.0

The pytorch/pytorch docker base image was used rather than [NVIDIA NGC container 24.12](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-12.html)
because the NGC container relies on an early release version of Torch-TensorRT 2.6.0a0 that introduced a [bug](https://github.com/pytorch/TensorRT/issues/3226) that was later fixed with the latest 2.6.0 release. It was cleaner to install TensorRT and Torch-TensorRT in a new pytorch/pytorch base image, rather than updating the librariers in the nvcr.io/nvidia/pytorch:24.12-py3 NGC container.   

```
sudo docker build \
--build-arg CACHE_BUSTER=$(date +%s) \
--build-arg CUDA_ARCH="89" \
--build-arg TORCH_CUDA_ARCH_LIST="8.9" \
-t codetr:latest -f Dockerfile .
```

`CUDA_ARCH="89" TORCH_CUDA_ARCH_LIST="8.9"` correspond to the NVIDIA RTX 4090 GPU Compute Compatibility. Refer to this webpage https://developer.nvidia.com/cuda-gpus to update the numbers to match your GPU. You can also target multiple CUDA compute capabilities such as `CUDA_ARCH="76;89" TORCH_CUDA_ARCH_LIST="7.6;8.9"`


### 3. Run the container and run tests

```
sudo docker run \
--gpus device=0 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
-v /home/bryan/expr/co-detr/docker:/workspace/output \
-it --rm codetr:latest
```
You should specify the target GPU device `--gpus device=0` and you can update `/home/bryan/expr/co-detr/docker` to a directory on your local filesystem.

#### Run the test suite
```
pytest csrc_tests/test_plugin.py -s --plugin-lib codetr/csrc/build/libdeformable_attention_plugin.so
pytest tests/test_multi_scale_deformable_attention.py -s
pytest tests/test_export.py -s
```

### 4. Export the Co-DETR model from Pytorch to TensorRT and run inference in C++

The [co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth) weights are automatically downloaded with the Dockerfile build and saved to `/workspace`.

```
python export.py \
--dtype float16 \
--optimization-level 3 \
--output /workspace/output \
--height 768 --width 1152 \
--weights /workspace/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth \
--plugin-lib codetr/csrc/build/libdeformable_attention_plugin.so
```
see [export.py](export.py) for optional arguments


Run inference C++ using the compiled TorchScript model 
```
cd build
./codetr_inference \
--model /workspace/output/codetr.ts \
--input ../assets/demo.jpg \
--output /workspace/output/cpp_ts_output.jpg \
--benchmark-iterations 100 \
--trt-plugin-path ../codetr/csrc/build/libdeformable_attention_plugin.so \
--dtype float16 --target-height 768 --target-width 1152
```

Run inference in C++ using the compile TensorRT serialized engine file. Notice, there is NO dependency on the Torch-TensorRT C++ library when executing the TensorRT serialized engine file.
```
./codetr_inference \
--model /workspace/output/codetr.engine \
--input ../assets/demo.jpg \
--output /workspace/output/cpp_engine_output.jpg \
--benchmark-iterations 100 \
--trt-plugin-path ../codetr/csrc/build/libdeformable_attention_plugin.so \
--dtype float16 --target-height 768 --target-width 1152
```

Note
* The Co-DETR model is exported to TensorRT with a fixed input height and width because exporting with dynamic shapes takes an extremely long time. You can read more about dynamic shapes with Torch-TensorRT [here](https://pytorch.org/TensorRT/user_guide/dynamic_shapes.html).
* The Swin Transformer backbone downscales the input image by a factor of 32x.

The `/workspace/output` directory will look like

<img src="https://github.com/user-attachments/assets/6171aa55-f7f0-4fe2-8422-e8bdd47831a9" width="300"/>

### Results: PyTorch (python) vs. TensorRT (python) detections vs. TensorRT (C++) detections
<img src="https://github.com/user-attachments/assets/1b1529be-df73-4b4d-8ff6-c5a4ac4c7a05" width="300"/>
<img src="https://github.com/user-attachments/assets/e2054c85-aeb3-44fd-af29-3c440ab7e95a" width="300"/>
<img src="https://github.com/user-attachments/assets/85de858f-3734-48a1-927c-99b872f730af" width="300"/>

# Benchmarking with trtexec

The TensorRT inference runtime can be benchmarked using `trtexec`
```
LD_PRELOAD=./codetr/csrc/build/libdeformable_attention_plugin.so \
trtexec \
--loadEngine=/workspace/output/codetr.engine \
--fp16 --useSpinWait --useCudaGraph \
--iterations=100 --warmUp=500 --avgRuns=100  \
> /workspace/output/trtexec-benchmark.log 2>&1
```
The runtime performance is summarized at the bottom of the output log
```
...
[04/11/2025-15:15:48] [I] TensorRT version: 10.7.0
[04/11/2025-15:15:48] [I] Loading standard plugins
[04/11/2025-15:15:48] [I] [TRT] Loaded engine size: 472 MiB
[04/11/2025-15:15:48] [I] Engine deserialized in 0.212037 sec.
[04/11/2025-15:15:48] [I] [TRT] [MS] Running engine with multi stream info
[04/11/2025-15:15:48] [I] [TRT] [MS] Number of aux streams is 3
[04/11/2025-15:15:48] [I] [TRT] [MS] Number of total worker streams is 4
[04/11/2025-15:15:48] [I] [TRT] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[04/11/2025-15:15:48] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +1025, now: CPU 0, GPU 1484 (MiB)
[04/11/2025-15:15:48] [I] Setting persistentCacheLimit to 0 bytes.
[04/11/2025-15:15:48] [I] Created execution context with device memory size: 1024.62 MiB
[04/11/2025-15:15:48] [I] Using random values for input batch_inputs
[04/11/2025-15:15:48] [I] Input binding for batch_inputs with dimensions 1x3x768x1152 is created.
[04/11/2025-15:15:48] [I] Using random values for input img_masks
[04/11/2025-15:15:48] [I] Input binding for img_masks with dimensions 1x768x1152 is created.
[04/11/2025-15:15:48] [I] Output binding for output0 with dimensions 1x300x4 is created.
[04/11/2025-15:15:48] [I] Output binding for output1 with dimensions 1x300 is created.
[04/11/2025-15:15:48] [I] Output binding for output2 with dimensions 1x300 is created.
[04/11/2025-15:15:48] [I] Starting inference
[04/11/2025-15:15:48] [I] Capturing CUDA graph for the current execution context
[04/11/2025-15:15:48] [I] Successfully captured CUDA graph for the current execution context
[04/11/2025-15:15:52] [I] Warmup completed 14 queries over 500 ms
[04/11/2025-15:15:52] [I] Timing trace has 101 queries over 3.08403 s
[04/11/2025-15:15:52] [I] 
[04/11/2025-15:15:52] [I] === Trace details ===
[04/11/2025-15:15:52] [I] Trace averages of 100 runs:
[04/11/2025-15:15:52] [I] Average on 100 runs - GPU latency: 30.211 ms - Host latency: 30.8067 ms (enqueue 0.00390778 ms)
[04/11/2025-15:15:52] [I] 
[04/11/2025-15:15:52] [I] === Performance summary ===
[04/11/2025-15:15:52] [I] Throughput: 32.7494 qps
[04/11/2025-15:15:52] [I] Latency: min = 30.7003 ms, max = 31.6415 ms, mean = 30.8066 ms, median = 30.7695 ms, percentile(90%) = 30.8008 ms, percentile(95%) = 31.1944 ms, percentile(99%) = 31.5444 ms
[04/11/2025-15:15:52] [I] Enqueue Time: min = 0.00305176 ms, max = 0.0141602 ms, mean = 0.00393677 ms, median = 0.00360107 ms, percentile(90%) = 0.00494385 ms, percentile(95%) = 0.00585938 ms, percentile(99%) = 0.00842285 ms
[04/11/2025-15:15:52] [I] H2D Latency: min = 0.5896 ms, max = 0.614014 ms, mean = 0.590921 ms, median = 0.590332 ms, percentile(90%) = 0.591431 ms, percentile(95%) = 0.592529 ms, percentile(99%) = 0.602051 ms
[04/11/2025-15:15:52] [I] GPU Compute Time: min = 30.1056 ms, max = 31.0457 ms, mean = 30.2108 ms, median = 30.1722 ms, percentile(90%) = 30.2061 ms, percentile(95%) = 30.6002 ms, percentile(99%) = 30.9484 ms
[04/11/2025-15:15:52] [I] D2H Latency: min = 0.00390625 ms, max = 0.00634766 ms, mean = 0.00490275 ms, median = 0.00463867 ms, percentile(90%) = 0.00585938 ms, percentile(95%) = 0.00598145 ms, percentile(99%) = 0.00628662 ms
[04/11/2025-15:15:52] [I] Total Host Walltime: 3.08403 s
[04/11/2025-15:15:52] [I] Total GPU Compute Time: 3.05129 s
```
Runtime measurement definitions:
* GPU Compute Time: the GPU latency to execute the kernels for a query.
* Total GPU Compute Time: the summation of the GPU Compute Time of all the queries.
* Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.
* H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.
* D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.
* Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency of a single query. Lower latency is better.
* Throughput: how many inferences can be completed in a fixed time. Higher throughput is better. Higher throughput reflects a more efficient utilization of fixed compute resources. 

# Profiling with trtexec and NVIDIA Nsight Systems

[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) can be used to profile the Co-DETR TensorRT engine execution. TensorRT uses [NVIDIA Took Extension SDK (NVTX)](https://docs.nvidia.com/nsight-visual-studio-edition/2020.1/nvtx/index.html) to record the start and stop timestamps for each layer. By prefixing the `trtexec` command with `nsys profile`, we can record the timing events and to a log file for visualization and analysis in the Nsight Systems application.

```
LD_LIBRARY_PATH=/home/bryan/src/libtorch/lib:/home/bryan/src/TensorRT-10.7.0.23/lib:$LD_LIBRARY_PATH \
LD_PRELOAD=./codetr/csrc/build/libdeformable_attention_plugin.so  \
nsys profile -o nsys_profiler \
--capture-range cudaProfilerApi --cuda-memory-usage=true \
trtexec \
--loadEngine=codetr.engine \
--fp16 \
--iterations=100 --warmUp=500 --avgRuns=100 --useSpinWait
```
(`nsys` isn't installed in the Docker image. I ran `nsys` on my local PC.)

then run
```
> nsight-sys
```
![nsight-systems-kernel-summary](https://github.com/user-attachments/assets/d5af891e-db99-4947-b9f2-d216a45652f3)

In Nsight Systems, you can open the `nsys_profiler.nsys-rep` file to visualize `trtexec` execution and trace TensorRT kernel calls. In the Timeline View, a single `ExecutionContext::enqueue` call (which runs the Co-DETR model) took 32.772 ms. The Stats panel’s CUDA GPU Kernel Summary shows detailed timing, where the `codetr::ms_deformable_im2col_gpu_kernel`—the core of the multiscale deformable attention operator—averaged 368.936 µs per inference, accounting for 6.0% of the total runtime.

If `trtexec` is run with `--useCudaGraph` then the the first enqueue will capture the CUDA graph (including all kernels, memory transfers, etc) and subsequent iterations will be extremely fast. In this case `--cuda-graph-trace=node` flag should be added to the nsys command to see the per-kernel runtime information.


By default, TensorRT only shows layer names in the NVTX markers. torch_tensorrt builds the TensorRT engine with profiling verbosity set to ProfilingVerbosity::kLAYER_NAMES_ONLY, which records the layer names, execution time per layer, and layer order in the engine. Unfortunately torch_tensorrt compilation doesn't provide an option to build the engine with ProfilingVerbosity::kDETAILED which would expose detailed layer information including tensor input/output names, shapes and data types, tensor formats, chosen tactics, and memory usage per layer.


References
* https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html#performance-benchmarking-using-trtexec
* NVTX https://docs.nvidia.com/nsight-visual-studio-edition/2020.1/nvtx/index.html
* NVTX Trace https://docs.nvidia.com/nsight-systems/UserGuide/index.html#nvtx-trace


# [Developer Tips] - PyTorch C++ CUDA Extensions
The C++ CUDA implementation [codetr/csrc/ms_deform_attn.cu](codetr/csrc/ms_deform_attn.cu) of the multi-scale deformable attention operator (a key component to the Deformable DETR architecture) is registered with PyTorch using the `TORCH_LIBRARY` macro. It is important to define the `TORCH_LIBRARY` registration in a separate file [codetr/csrc/deformable_attention_torch.cpp](codetr/csrc/deformable_attention_torch.cpp) to avoid polluting the C++ CUDA definition [codetr/csrc/ms_deform_attn.cu](codetr/csrc/ms_deform_attn.cu) with a `#include <torch/library.h>` import.

To add `torch.compile` support for the newly registered `codetr::multi_scale_deformable_attention` operator, we must add a FakeTensor kernel (also known as a "meta kernel" or "abstract impl"). FakeTensors are Tensors that have metadata (such as shape, dtype, device) but no data: the FakeTensor kernel for an operator specifies how to compute the metadata of output tensors given the metadata of input tensors. See [`def _multi_scale_deformable_attention_fake()` in codetr/ops.py](codetr/ops.py) for the specifics.

See the [setup.py](setup.py) for how to build the C++ CUDA Extension.

References
- https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial
- https://github.com/pytorch/vision/tree/main/torchvision/csrc


# [Developer Tips] - Implementing a TensorRT Custom Operator
Registering the custom C++ CUDA operator enables us to run Co-DETR in python. However, when the model is compiled to TensorRT (via `torch.compile(model, backend="tensorrt")`) we find that the compiled graph contains an alternating pattern of TorchTensorRTModule and GraphModules. `torch.compile` doesn't know how to convert our custom operator `codetr::multi_scale_deformable_attention` to TensorRT, so it is executed in PyTorch, which results in a hybrid execution graph. Switching between TensorRT and PyTorch execution slows down inference time substantially.

To enable full TensorRT compilation, we have to implement a TensorRT C++ Custom Plugin (`DeformableAttentionPlugin`) and register it with torch dynamo so that Torch-TensorRT can replace the `codetr::multi_scale_deformable_attention` operator with the `DeformableAttentionPlugin` at `torch.compile` time. 

The following three steps to implement and register a TensorRT C++ Custom Plugin are defined in [codetr/csrc/deformable_attention_plugin.cpp](codetr/csrc/deformable_attention_plugin.cpp).
1. Implement a plugin class (`DeformableAttentionPlugin`) derived from TensorRT’s plugin base classes (`IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild`, `IPluginV3OneRuntime`).
2. Implement a plugin creator class (`DeformableAttentionPluginCreator`) tied to our plugin class by deriving from TensorRT’s plugin creator base class (`IPluginCreatorV3One`).
3. Register the plugin creator class with TensorRT’s plugin registry (`REGISTER_TENSORRT_PLUGIN(DeformableAttentionPluginCreator);`)

The `DeformableAttentionPlugin` class is registered with torch dynamo in [codetr/ops.py](codetr/ops.py).
```
@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(torch.ops.codetr.multi_scale_deformable_attention.default)
def multi_scale_deformable_attention_converter(
    ctx: torch_tensorrt.dynamo.conversion.ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> trt.ITensor:
```

References:
* A useful guide providing examples for IPluginV3 and IPluginV3OneBuild https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html
* https://github.com/leimao/TensorRT-Custom-Plugin-Example
* https://github.com/NVIDIA/TensorRT/tree/release/10.0/samples/python/python_plugin
* Example plugins https://github.com/NVIDIA/TensorRT/tree/main/plugin#tensorrt-plugins

Documentation
* TensorRT python API documentation https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/index.html
* TensorRT C++ Documentation https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/index.html
* TensorRT C++ Documentation https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html

Dynamo References
* https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/custom_kernel_plugins.html
* https://pytorch.org/TensorRT/contributors/dynamo_converters.html
*  torch_tensorrt dynamo/conversion library https://github.com/pytorch/TensorRT/tree/v2.6.0/py/torch_tensorrt/dynamo/conversion


# [Developer Tips] - code formatting

```
isort codetr
flake8 codetr
black codetr
clang-format -i <path-to-C++-file>
```

# Acknowledgments

This project includes code adapted from [MMDetection](https://github.com/open-mmlab/mmdetection),
licensed under the Apache License 2.0.