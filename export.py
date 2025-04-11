import os
import sys
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np

import torch
import torch_tensorrt
from torch.utils import benchmark
from torchvision.ops import batched_nms

import mmcv
from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer

from codetr import build_CoDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Export CoDETR model to TensorRT")
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py"),
        help="Path to model config file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/bryan/expr/co-detr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--image", type=str, default=os.path.join(PROJECT_ROOT, "assets/demo.jpg"), help="Path to test image"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Precision for TensorRT compilation",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="TensorRT optimization level (0-5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="codetr_tensorrt",
        help="Output directory to save results including the exported TensorRT model and "
        "the input image with predicted boxes overlayed. "
        "PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file.",
    )
    parser.add_argument("--height", type=int, default=768, help="Input height")
    parser.add_argument("--width", type=int, default=1152, help="Input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    # Benchmarking
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run for benchmarking",
    )
    # Visualization
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold for filtering predictions",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used in non-maximum suppression for suppressing false positive detections",
    )
    parser.add_argument(
        "--plugin-lib",
        type=str,
        default=os.path.join(PROJECT_ROOT, "codetr/csrc/build/libdeformable_attention_plugin.so"),
        help="Path to the plugin library",
    )
    parser.add_argument(
        "--ignore-aspect-ratio",
        action="store_true",
        help="Ignore aspect ratio when resizing the input image",
    )
    return parser.parse_args()


def get_test_pipeline(height=768, width=1152, ignore_aspect_ratio=False):
    return Compose(
        [
            dict(type="mmdet.InferencerLoader"),
            dict(type="mmdet.Resize", scale=(width, height), keep_ratio=not ignore_aspect_ratio),
            dict(type="mmdet.Pad", size=(width, height)),
            dict(
                type="mmdet.PackDetInputs", meta_keys=("ori_shape", "img_shape", "scale_factor", "img_unpadded_shape")
            ),
        ]
    )


def preprocess_image(image_array, cfg, height, width, batch_size=1, ignore_aspect_ratio=False):
    pipeline = get_test_pipeline(height, width, ignore_aspect_ratio)
    data_preprocessor_cfg = cfg.model.data_preprocessor
    img_mean = data_preprocessor_cfg.get("mean", [123.675, 116.28, 103.53])
    img_std = data_preprocessor_cfg.get("std", [58.395, 57.12, 57.375])
    img_mean = torch.tensor(img_mean).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std).view(1, 3, 1, 1)

    data = pipeline(image_array)
    image_tensor = data["inputs"].to(torch.float32)  # (3,height,width)
    data_sample = data["data_samples"]

    batch_inputs = image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size,3,height,width)
    batch_inputs = (batch_inputs - img_mean) / img_std

    # 0 within image, 1 in padded region
    img_masks = torch.ones((batch_size, height, width), dtype=batch_inputs.dtype)
    unpad_h, unpad_w = data_sample.metainfo.get("img_unpadded_shape", (height, width))
    img_masks[:, :unpad_h, :unpad_w] = 0
    return batch_inputs, img_masks, data_sample


def benchmark_runtime(run_pytorch_model, run_tensorrt_model, iterations=10):
    with torch.inference_mode():
        # Warm-up runs
        _ = run_pytorch_model()
        _ = run_tensorrt_model()

        t0 = benchmark.Timer(
            stmt="run_pytorch_model()",
            globals={"run_pytorch_model": run_pytorch_model},
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="run_tensorrt_model()",
            globals={"run_tensorrt_model": run_tensorrt_model},
            num_threads=1,
        )

        # Run benchmarks
        pytorch_time = t0.timeit(iterations)
        tensorrt_time = t1.timeit(iterations)

        print(f"\nPyTorch implementation: {pytorch_time}")
        print(f"TensorRT implementation: {tensorrt_time}")

        # Calculate speedups
        speedup_trt = pytorch_time.mean / tensorrt_time.mean

        print(f"\nTensorRT speedup: {speedup_trt:.2f}x\n")


class Visualizer:
    def __init__(
        self,
        image_array: np.ndarray,
        data_sample: DetDataSample,
        dataset_meta: Dict[str, Any],
        score_threshold: float = 0.0,
        iou_threshold: float = 0.8,
    ):
        self.image_array = image_array
        self.data_sample = data_sample
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.visualizer = DetLocalVisualizer(vis_backends=[{"type": "LocalVisBackend", "save_dir": None}])
        self.visualizer.dataset_meta = dataset_meta

    def postprocess_predictions(
        self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        if self.score_threshold > 0:
            valid_mask = scores > self.score_threshold
            # TODO: handle case where valid_mask is all False
            scores = scores[valid_mask]
            boxes = boxes[valid_mask]
            labels = labels[valid_mask]

        keep_idxs = batched_nms(boxes, scores, labels, self.iou_threshold)
        boxes = boxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

        # rescale to the original image size
        scale_factor = self.data_sample.metainfo["scale_factor"]
        boxes /= boxes.new_tensor(scale_factor).repeat((1, 2))

        results = InstanceData()
        results.bboxes = boxes
        results.scores = scores
        results.labels = labels

        return results

    def __call__(self, batch_boxes: torch.Tensor, batch_scores: torch.Tensor, batch_labels: torch.Tensor):
        # only visualize the first image
        postprocess_predictions = self.postprocess_predictions(batch_boxes[0], batch_scores[0], batch_labels[0])
        data_sample = self.data_sample.clone()
        data_sample.pred_instances = postprocess_predictions

        self.visualizer.add_datasample(
            "image",
            self.image_array,
            data_sample=data_sample,
            draw_gt=False,
            draw_pred=True,
            show=False,
            pred_score_thr=self.score_threshold,
        )
        vis = self.visualizer.get_image()
        return vis


def main():
    args = parse_args()

    # Disable TensorRT safe mode to avoid warnings
    torch_tensorrt.runtime.set_multi_device_safe_mode(False)

    # Convert dtype string to torch dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    device = "cuda:0"
    print(f"Loading model from {args.model}")
    if args.weights:
        print(f"Loading weights from {args.weights}")

    # Build CoDETR model
    model, dataset_meta = build_CoDETR(args.model, args.weights, device)
    model.to(dtype)
    model.eval()

    # Get model config for preprocessing
    cfg = Config.fromfile(args.model)

    # Load image
    print(f"Loading image from {args.image}")
    image_array = mmcv.imread(args.image, channel_order="rgb")

    # Preprocess image to get proper inputs
    batch_inputs, img_masks, data_sample = preprocess_image(
        image_array, cfg, args.height, args.width, args.batch_size, args.ignore_aspect_ratio
    )
    visualizer = Visualizer(image_array, data_sample, dataset_meta, args.score_threshold, args.iou_threshold)

    batch_inputs = batch_inputs.to(dtype).to(device)
    img_masks = img_masks.to(dtype).to(device)

    print(f"Input tensor shape: {batch_inputs.shape}")
    print(f"Mask tensor shape: {img_masks.shape}")

    # Export model to ExportedProgram and then to TensorRT
    print(f"Exporting model with precision={args.dtype}, optimization_level={args.optimization_level}")
    os.makedirs(args.output, exist_ok=True)
    with torch.inference_mode():

        def run_pytorch_model():
            return model(batch_inputs, img_masks)

        output_pytorch = run_pytorch_model()

        vis = visualizer(*output_pytorch)
        vis_path = os.path.join(args.output, "pytorch_output.jpg")
        mmcv.imwrite(mmcv.rgb2bgr(vis), vis_path)
        print(f"✅ Pytorch predictions visualized to {vis_path}")

        # First export to ExportedProgram
        model_export = torch.export.export(
            model,
            args=(batch_inputs, img_masks),
            strict=True,
        )
        print(f"✅ Model exported to ExportedProgram")

        # Then compile with TensorRT
        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=args.optimization_level,
            truncate_double=True,
            require_full_compilation=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
            use_python_runtime=False,
        )
        print(f"✅ Model compiled to TensorRT at optimization level: {args.optimization_level}")
        model_type = torch_tensorrt._compile._parse_module_type(model_trt)
        print(f"TensorRT model type: {type(model_trt)}\t {model_type}")

        def run_tensorrt_model():
            return model_trt(batch_inputs, img_masks)

        # Test inference
        output_trt = run_tensorrt_model()

        print(f"Test inference successful! Output shapes:")
        print(f"  boxes: {output_trt[0].shape}")
        print(f"  scores: {output_trt[1].shape}")
        print(f"  labels: {output_trt[2].shape}")

        vis = visualizer(*output_trt)
        vis_path = os.path.join(args.output, "tensorrt_output.jpg")
        mmcv.imwrite(mmcv.rgb2bgr(vis), vis_path)
        print(f"✅ TensorRT predictions visualized to {vis_path}")

        torch.cuda.empty_cache()
        benchmark_runtime(run_pytorch_model, run_tensorrt_model, iterations=args.iterations)
        torch.cuda.empty_cache()
        save_model(os.path.join(args.output, "codetr.ts"), model_trt, (batch_inputs, img_masks))
        print_tensorrt_model(model_trt, os.path.join(args.output, "tensorrt_model.txt"))

        # Then compile with TensorRT
        model_trt_engine_bytes = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
            model_export,
            inputs=(batch_inputs, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=args.optimization_level,
            truncate_double=True,
            require_full_compilation=True,
            use_python_runtime=False,
        )
        with open(os.path.join(args.output, "codetr.engine"), "wb") as f:
            f.write(model_trt_engine_bytes)


def save_model(save_path, model, inputs):
    """
    for 'torchscript' output format, this should be equivalent to

    If the model is torch.fx.GraphModule, then first it will be retraced then saved
    model_ts = torch.jit.trace(model_trt, inputs)
    torch.jit.save(model_ts, save_path)

    If the model is a torch.jit.TopLevelTracedModule (torchscript), then it will be saved directly.
    torch.jit.save(model, save_path)
    """
    output_format = "exported_program" if save_path.endswith(".ep") else "torchscript"
    print(f"Saving TensorRT model to {save_path}")
    torch_tensorrt.save(model, save_path, inputs=inputs, output_format=output_format)
    print(f"✅ Model saved successfully")


def print_tensorrt_model(model, save_path):
    with open(save_path, "w") as f:
        f.write("TensorRT model structure:\n")
        f.write(str(model))
        f.write("\n\nTensorRT model structure [DEBUG MODE]:\n")
        f.write(model.print_readable())
    print(f"✅ TensorRT model structure saved to {save_path}")


if __name__ == "__main__":
    main()
