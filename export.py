import os
import torch
import torch_tensorrt
import argparse

import mmcv
from mmengine.dataset import Compose
from mmengine.config import Config

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
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to run for benchmarking",
    )
    return parser.parse_args()


def get_test_pipeline(height=768, width=1152):
    return Compose(
        [
            dict(type="mmdet.InferencerLoader"),
            dict(type="mmdet.Resize", scale=(width, height), keep_ratio=True),
            dict(type="mmdet.Pad", size=(width, height)),
            dict(
                type="mmdet.PackDetInputs", meta_keys=("ori_shape", "img_shape", "scale_factor", "img_unpadded_shape")
            ),
        ]
    )


def preprocess_image(image_array, cfg, height, width, batch_size=1):
    pipeline = get_test_pipeline(height, width)
    data_preprocessor_cfg = cfg.model.data_preprocessor
    img_mean = data_preprocessor_cfg.get("mean", [123.675, 116.28, 103.53])
    img_std = data_preprocessor_cfg.get("std", [58.395, 57.12, 57.375])
    img_mean = torch.tensor(img_mean).view(1, 3, 1, 1)
    img_std = torch.tensor(img_std).view(1, 3, 1, 1)

    data = pipeline(image_array)
    image_tensor = data["inputs"].to(torch.float32)  # (3,height,width)
    data_samples = data["data_samples"]

    batch_inputs = image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size,3,height,width)
    batch_inputs = (batch_inputs - img_mean) / img_std

    # 0 within image, 1 in padded region
    img_masks = torch.ones((batch_size, height, width), dtype=batch_inputs.dtype)
    unpad_h, unpad_w = data_samples.metainfo.get("img_unpadded_shape", (height, width))
    img_masks[:, :unpad_h, :unpad_w] = 0
    return batch_inputs, img_masks


def benchmark_runtime(run_pytorch_model, run_tensorrt_model, iterations=10):
    with torch.inference_mode():
        # Warm-up runs
        _ = run_pytorch_model()
        _ = run_tensorrt_model()

        t0 = torch.benchmark.Timer(
            stmt="run_pytorch_model()",
            globals={"run_pytorch_model": run_pytorch_model},
            num_threads=1,
        )

        t1 = torch.benchmark.Timer(
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

        print(f"\nTensorRT speedup: {speedup_trt:.2f}x")


def main():
    args = parse_args()

    # Convert dtype string to torch dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    # Disable TensorRT safe mode to avoid warnings
    torch_tensorrt.runtime.set_multi_device_safe_mode(False)

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
    batch_inputs, img_masks = preprocess_image(image_array, cfg, args.height, args.width, args.batch_size)
    batch_inputs = batch_inputs.to(dtype).to(device)
    img_masks = img_masks.to(dtype).to(device)

    print(f"Input tensor shape: {batch_inputs.shape}")
    print(f"Mask tensor shape: {img_masks.shape}")

    # Export model to ExportedProgram and then to TensorRT
    print(f"Exporting model with precision={args.dtype}, optimization_level={args.optimization_level}")
    with torch.inference_mode():

        def run_pytorch_model():
            return model(batch_inputs, img_masks)

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
        )
        print(f"✅ Model compiled to TensorRT at optimization level: {args.optimization_level}")

        def run_tensorrt_model():
            return model_trt(batch_inputs, img_masks)

        # Test inference
        output_trt = run_tensorrt_model()

        print(f"Test inference successful! Output shapes:")
        print(f"  boxes: {output_trt[0].shape}")
        print(f"  scores: {output_trt[1].shape}")
        print(f"  labels: {output_trt[2].shape}")

        torch.cuda.empty_cache()
        benchmark_runtime(run_pytorch_model, run_tensorrt_model, run_tensorrt_model, iterations=args.iterations)
        torch.cuda.empty.cache()
        # Save the model if requested
        os.makedirs(args.output, exist_ok=True)
        save_model(os.path.join(args.output, "codetr.ep"), model_export, (batch_inputs, img_masks))
        save_model(os.path.join(args.output, "codetr.ts"), model_trt, (batch_inputs, img_masks))


def save_model(save_path, model, inputs):
    output_format = "exported_program" if save_path.endswith(".ep") else "torchscript"
    print(f"Saving TensorRT model to {save_path}")
    torch_tensorrt.save(model, save_path, inputs=inputs, output_format=output_format)
    print(f"✅ Model saved successfully")


if __name__ == "__main__":
    main()
