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
        help="Output file for TensorRT model (if saving). Both the 'exported_program' .ep and 'torchscript' .ts "
        "output formats will be saved.  PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file.",
    )
    parser.add_argument("--save", action="store_true", help="Save compiled model to disk")
    parser.add_argument("--height", type=int, default=768, help="Input height")
    parser.add_argument("--width", type=int, default=1152, help="Input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    return parser.parse_args()


def get_test_pipeline(height=768, width=1152):
    return Compose(
        [
            dict(type="mmdet.InferencerLoader"),
            dict(type="Resize", scale=(width, height), keep_ratio=True),
            dict(type="Pad", size=(width, height)),
            dict(type="PackDetInputs", meta_keys=("ori_shape", "img_shape", "scale_factor", "img_unpadded_shape")),
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
    detr_model, dataset_meta = build_CoDETR(args.model, args.weights, device)
    detr_model.to(dtype)
    detr_model.eval()

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

    # Export model to TorchScript and then to TensorRT
    print(f"Exporting model with precision={args.dtype}, optimization_level={args.optimization_level}")
    with torch.inference_mode():
        output_pytorch = detr_model(batch_inputs, img_masks)

        # First export to TorchScript
        model_export = torch.export.export(
            detr_model,
            args=(batch_inputs, img_masks),
            strict=True,
        )
        print(f"✅ Model exported to TorchScript")

        # Then compile with TensorRT
        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=args.optimization_level,
        )
        print(f"✅ Model compiled to TensorRT at optimization level: {args.optimization_level}")

        # Test inference
        output_trt = model_trt(batch_inputs, img_masks)

    print(f"Test inference successful! Output shapes:")
    print(f"  boxes: {output_trt[0].shape}")
    print(f"  scores: {output_trt[1].shape}")
    print(f"  labels: {output_trt[2].shape}")

    for i, name in enumerate(["boxes", "scores", "labels"]):
        abs_diff = torch.abs(output_pytorch[i] - output_trt[i])
        top_5_diff, top_5_locations = torch.topk(abs_diff.flatten(), 5)
        print(f"Top 5 absolute differences for {name}: {top_5_diff.tolist()} at locations {top_5_locations.tolist()}")

    # Save the model if requested
    if args.save:
        output_path = os.path.splitext(args.output)[0]
        print(f"Saving TensorRT model to {output_path}.ep")
        torch_tensorrt.save(
            model_trt, output_path + ".ep", inputs=(batch_inputs, img_masks), output_format="exported_program"
        )
        print(f"✅ Model saved successfully")
        print(f"Saving TensorRT model to {output_path}.ts")
        torch_tensorrt.save(
            model_trt, output_path + ".ts", inputs=(batch_inputs, img_masks), output_format="torchscript"
        )
        print(f"✅ Model saved successfully")


if __name__ == "__main__":
    main()
