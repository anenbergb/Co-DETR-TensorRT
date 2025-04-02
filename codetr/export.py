import os
import torch
import torch_tensorrt
import argparse
import mmcv
import numpy as np

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.config import Config

from codetr.codetr import build_CoDETR
from codetr.inferencer import Inferencer


def parse_args():
    parser = argparse.ArgumentParser(description="Export CoDETR model to TensorRT")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
        "--image", type=str, default="/home/bryan/src/mmdetection/demo/demo.jpg", help="Path to test image"
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
        "--output", type=str, default="codetr_tensorrt.pt", help="Output file for TensorRT model (if saving)"
    )
    parser.add_argument("--save", action="store_true", help="Save compiled model to disk")
    parser.add_argument("--height", type=int, default=768, help="Input height")
    parser.add_argument("--width", type=int, default=1152, help="Input width")
    return parser.parse_args()


def preprocess_image(img, cfg, device, input_height, input_width):
    """Preprocess image using model's data preprocessor"""
    # Configure data preprocessor from model config
    data_preprocessor_cfg = cfg.model.data_preprocessor.copy()
    assert data_preprocessor_cfg.pop("type") == "DetDataPreprocessor"
    data_preprocessor = DetDataPreprocessor(
        mean = data_preprocessor_cfg.pop("mean"),
        std = data_preprocessor_cfg.pop("std"),
        
    ).to(device)
    import ipdb; ipdb.set_trace()


    # Create data dict like what would be produced by pipeline and collate_fn
    data = {
        "inputs": [torch.from_numpy(img.transpose(2, 0, 1)).float()],
        "data_samples": [
            {
                "metainfo": {
                    "img_shape": img.shape[:2],
                    "img_unpadded_shape": img.shape[:2],  # Original image size
                    "scale_factor": np.ones(4),  # No scaling initially
                }
            }
        ],
    }

    # Process data through the data preprocessor
    data_processed = data_preprocessor(data, False)
    batch_inputs = data_processed["inputs"].to(device)
    batch_data_samples = data_processed["data_samples"]

    # Create img_masks tensor (0 in image area, 1 in padded region)
    bs, _, H, W = batch_inputs.shape
    img_masks = torch.ones((bs, H, W), device=device, dtype=batch_inputs.dtype)
    for i, data_samples in enumerate(batch_data_samples):
        unpad_h, unpad_w = data_samples.metainfo.get("img_unpadded_shape", (H, W))
        img_masks[i, :unpad_h, :unpad_w] = 0

    return batch_inputs, img_masks, batch_data_samples


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
    batch_inputs, img_masks, _ = preprocess_image(image_array, cfg, device, args.height, args.width)
    batch_inputs = batch_inputs.to(dtype)
    img_masks = img_masks.to(dtype)
    
    import ipdb
    ipdb.set_trace()
    print(f"Input tensor shape: {batch_inputs.shape}")
    print(f"Mask tensor shape: {img_masks.shape}")

    # Export model to TorchScript and then to TensorRT
    print(f"Exporting model with precision={args.dtype}, optimization_level={args.optimization_level}")
    with torch.inference_mode():
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
        print(f"✅ Model compiled to TensorRT")

        # Test inference
        predictions = model_trt(batch_inputs, img_masks)
        print(f"Test inference successful! Output shapes:")
        print(f"  boxes: {predictions[0].shape}")
        print(f"  scores: {predictions[1].shape}")
        print(f"  labels: {predictions[2].shape}")

        # Save the model if requested
        if args.save:
            output_path = args.output
            print(f"Saving TensorRT model to {output_path}")
            torch_tensorrt.save(model_trt, output_path)
            print(f"✅ Model saved successfully")


if __name__ == "__main__":
    main()
