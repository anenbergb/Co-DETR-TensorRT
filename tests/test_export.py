import torch
import pytest

from codetr.swin import SwinTransformer
from codetr.co_dino_head import CoDINOHead

from mmengine.config import Config
from mmdet.registry import MODELS

import torch_tensorrt

from .helpers import benchmark_runtime

torch_tensorrt.runtime.set_multi_device_safe_mode(False)

# model settings
swin_config = dict(
    pretrain_img_size=384,
    embed_dims=192,
    depths=[2, 2, 18, 2],
    num_heads=[6, 12, 24, 48],
    window_size=12,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    # Please only add indices that would be used
    # in FPN, otherwise some parameter will not be used
    with_cp=True,
    convert_weights=True,
)
neck_config = {
    "type": "ChannelMapper",
    "in_channels": [192, 384, 768, 1536],
    "kernel_size": 1,
    "out_channels": 256,
    "act_cfg": None,
    "norm_cfg": {"type": "GN", "num_groups": 32},
    "num_outs": 5,
}

query_head_config = {
    #  'type': 'CoDINOHead',
    "num_query": 900,
    "num_classes": 80,
    "in_channels": 2048,
    "as_two_stage": True,
    "dn_cfg": {
        "label_noise_scale": 0.5,
        "box_noise_scale": 0.4,
        "group_cfg": {"dynamic": True, "num_groups": None, "num_dn_queries": 500},
    },
    "transformer": {
        "type": "CoDinoTransformer",
        "with_coord_feat": False,
        "num_co_heads": 2,
        "num_feature_levels": 5,
        "encoder": {
            "type": "DetrTransformerEncoder",
            "num_layers": 6,
            "with_cp": 6,
            "transformerlayers": {
                "type": "BaseTransformerLayer",
                "attn_cfgs": {
                    "type": "MultiScaleDeformableAttention",
                    "embed_dims": 256,
                    "num_levels": 5,
                    "dropout": 0.0,
                },
                "feedforward_channels": 2048,
                "ffn_dropout": 0.0,
                "operation_order": ("self_attn", "norm", "ffn", "norm"),
            },
        },
        "decoder": {
            "type": "DinoTransformerDecoder",
            "num_layers": 6,
            "return_intermediate": True,
            "transformerlayers": {
                "type": "DetrTransformerDecoderLayer",
                "attn_cfgs": [
                    {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8, "dropout": 0.0},
                    {"type": "MultiScaleDeformableAttention", "embed_dims": 256, "num_levels": 5, "dropout": 0.0},
                ],
                "feedforward_channels": 2048,
                "ffn_dropout": 0.0,
                "operation_order": ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            },
        },
    },
    "positional_encoding": {"type": "SinePositionalEncoding", "num_feats": 128, "temperature": 20, "normalize": True},
    "loss_cls": {"type": "QualityFocalLoss", "use_sigmoid": True, "beta": 2.0, "loss_weight": 1.0},
    "loss_bbox": {"type": "L1Loss", "loss_weight": 5.0},
    "loss_iou": {"type": "GIoULoss", "loss_weight": 2.0},
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_swin_transformer(dtype):
    print(f"Testing SwinTransformer with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(swin_config)
    model = SwinTransformer(**cfg).to(device).to(dtype)
    model.init_weights()
    model.eval()

    batch_inputs = torch.randn(1, 3, 1280, 1920, dtype=dtype, device=device)

    def run_pytorch_model():
        return model(batch_inputs)

    with torch.inference_mode():
        model_export = torch.export.export(
            model,
            args=(batch_inputs,),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(batch_inputs)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs,),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    # Use different tolerances based on precision
    tol_export = 1e-3 if dtype == torch.float32 else 1e-3
    tol_trt = 1e-1 if dtype == torch.float32 else 1e-1

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_neck(dtype):
    print(f"Testing ChannelMapper neck with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(neck_config)
    model = MODELS.build(cfg).to(device).to(dtype)
    # model.init_weights()
    model.eval()

    input_height = 1280
    input_width = 1920
    batch_size = 1
    batch_inputs = []
    in_channels = cfg["in_channels"]
    downscales = [4, 8, 16, 32]
    for in_channel, downscale in zip(in_channels, downscales):
        batch_inputs.append(
            torch.randn(
                batch_size, in_channel, input_height // downscale, input_width // downscale, dtype=dtype, device=device
            )
        )

    batch_inputs = tuple(batch_inputs)

    def run_pytorch_model():
        return model(batch_inputs)

    with torch.inference_mode():
        model_export = torch.export.export(
            model,
            args=(batch_inputs,),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(batch_inputs)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs,),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    tol_export = 1e-3
    tol_trt = 1e-1

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_query_head(dtype):
    print(f"Testing CoDINOHead with dtype={dtype}")

    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    # cfg = Config(query_head_config)
    model = CoDINOHead(**query_head_config).to(device).to(dtype)
    model.init_weights()
    model.to(dtype)
    model.eval()

    pytest.set_trace()

    input_height = 1280
    input_width = 1920
    in_channels = 256
    batch_size = 1
    img_feats = []
    downscales = [4, 8, 16, 32, 64]
    for in_channel, downscale in zip(in_channels, downscales):
        img_feats.append(
            torch.randn(
                batch_size, in_channel, input_height // downscale, input_width // downscale, dtype=dtype, device=device
            )
        )
    # 0 within image, 1 in padded region
    # this is a dummy mask where all pixels are within the image
    img_masks = torch.zeros((1, input_height, input_width), device=device, dtype=dtype)

    def run_pytorch_model():
        return model(img_feats, img_masks)

    with torch.inference_mode():
        model_export = torch.export.export(
            model,
            args=(img_feats, img_masks),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)} with dtype={dtype}")

        def run_exported_model():
            return model_export.module()(img_feats, img_masks)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(img_feats, img_masks),
            enabled_precisions=(dtype,),
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT with dtype={dtype}")

        def run_tensorrt_model():
            return model_trt(img_feats, img_masks)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    tol_export = 1e-3
    tol_trt = 1e-1

    for i in range(len(output_pytorch)):
        if isinstance(output_pytorch[i], torch.Tensor):
            torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=tol_export, atol=tol_export)
            torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=tol_trt, atol=tol_trt)
        elif isinstance(output_pytorch[i], list):
            for j in range(len(output_pytorch[i])):
                torch.testing.assert_close(output_pytorch[i][j], output_export[i][j], rtol=tol_export, atol=tol_export)
                torch.testing.assert_close(output_pytorch[i][j], output_trt[i][j], rtol=tol_trt, atol=tol_trt)

    benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


if __name__ == "__main__":
    pytest.main([__file__])
