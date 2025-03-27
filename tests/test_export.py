import torch
import pytest

from codetr.swin import SwinTransformer
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

#     query_head=dict(
#         dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
#         transformer=dict(encoder=dict(with_cp=6)))
# )


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

    input_width = 1280
    input_height = 1920
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


if __name__ == "__main__":
    pytest.main([__file__])
